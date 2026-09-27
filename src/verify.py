"""
ubahn_verify.py - check every figure of an answer against the tool results that produced it.

    v = verify_answer(answer_text, agent.trace)
    v['summary']        -> {'checked': 24, 'verified': 21, 'calculated': 2, 'not_found': 1, 'not_found_values': ['5.655']}
    annotate_html(answer_text, v, prefix='m3')  -> Markdown with each figure wrapped in a clickable, colour-coded span
    sources_html(v, prefix='m3')                -> the numbered list of sources the figures link to
    summary_line(v)                             -> one line for the command line / batch output

A figure is "traced" (reported as verified in the data structures) when a tool result (or the arguments the model passed, e.g. an attendance assumption)
contains the same value up to the rounding shown in the answer; "calculated" when it is the difference, ratio,
sum or percentage change of two figures of the same tool object (e.g. 250 against 19 normally -> 13 times);
"not found" otherwise. Years, dates, line names (U8), list markers and small counts (0-10 without a unit) are
not checked.
"""
from __future__ import annotations

import html
import json
import re

MONTHS = ('january|february|march|april|may|june|july|august|september|october|november|december|'
          'janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|'
          'januar|februar|märz|juni|juli|oktober|dezember|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec')
APPROX = r'(?:about|around|roughly|approximately|approx\.?|nearly|almost|some|~|≈|environ|près de|presque|quasi|autour de|circa|ca\.)'

# a figure: optional sign, digits with optional thousands separators (, narrow nbsp, nbsp, thin space), optional decimals
NUM = re.compile(r'(?<![\w./:-])([-+−]?)(\d{1,3}(?:(?:,|\u202f|\u00a0|\u2009)\d{3})+|\d+)(?:([.,])(\d+))?(?![\w/])'
                 r'(\s?(?:%|×|x\b|times|fois|kWh|MWh|Wh|km|°C|mm/h|mm|h\b|min\b|s\b))?', re.I)
TIME = re.compile(r'(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?!\d)(?!:\d)')


# ------------------------------------------------------------------------------------------ answer side
def _spans_to_skip(text):
    """Code spans, URLs, ISO dates, headings' list numbers: figures inside them are not checked."""
    skip = []
    for pat in (r'`[^`]*`', r'https?://\S+', r'\b\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2})?\b', r'\b[US]\d{1,2}\b',
                r'(?m)^\s*(?:#+\s*)?\d+[.)]\s', r'\[\^?\d+\]', r'\b(?:19|20)\d{2}\b',
                r'\b15[\s-]?(?:min\w*|minutes?)\b', r'\bper\s+15\b', r'\b24[/ ]7\b',
                r'\b\d{1,3}\s*[–-]\s*\d{1,3}\s*(?:min\w*|minutes?)\b'):
        skip += [m.span() for m in re.finditer(pat, text)]
    for m in re.finditer(rf'\b\d{{1,2}}(?:st|nd|rd|th|er|e)?\s+(?:{MONTHS})\b|\b(?:{MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?\b', text, re.I):
        skip.append(m.span())
    return skip


def _inside(span, spans):
    return any(a <= span[0] and span[1] <= b for a, b in spans)


def _candidates(sign, intpart, sep, dec, unit, text_before):
    """Possible numeric readings of a displayed figure and the tolerance its rounding allows."""
    digits = re.sub(r'[,\u202f\u00a0\u2009]', '', intpart)
    grouped = digits != intpart
    unit = (unit or '').strip().lower()
    vals = []
    if dec is not None:
        if sep == ',' and not grouped and len(dec) == 3 and not unit.startswith(('%', 'kwh', 'mwh')):
            vals += [float(digits + dec), float(f'{digits}.{dec}')]       # "1,996": thousands (EN) or decimals (FR)
        else:
            vals.append(float(f'{digits}.{dec}'))
        ndec = len(dec)
    else:
        vals.append(float(digits))
        ndec = 0
    tol = 0.5 * 10 ** (-ndec) + 1e-9
    approx = re.search(APPROX + r'\s*\**\s*$', text_before, re.I) is not None
    v0 = vals[0]
    if ndec == 0 and v0 >= 100 and (approx or digits.endswith('0')):
        tol = max(tol, 0.012 * abs(v0))
    elif approx:
        tol *= 2
    negative = sign in ('-', '−')
    out = []
    for v in vals:
        v = -v if negative else v
        out.append((v, tol))
        if unit.startswith('%'):
            out += [(v / 100, tol / 100), (1 + v / 100, tol / 100), (1 - abs(v) / 100, tol / 100)]
    return out, tol, unit


def extract_figures(text: str):
    """Figures of an answer: [{'span', 'raw', 'values', 'tol', 'unit', 'kind'}]."""
    skip = _spans_to_skip(text)
    figs = []
    taken = []
    for m in TIME.finditer(text):
        if _inside(m.span(), skip):
            continue
        figs.append({'span': m.span(), 'raw': m.group(0), 'values': [], 'tol': 0, 'unit': '', 'kind': 'time',
                     'time': f'{int(m.group(1)):02d}:{m.group(2)}'})
        taken.append(m.span())
    for m in NUM.finditer(text):
        span = (m.start(), m.end(5) if m.group(5) else m.end(4) if m.group(4) else m.end(2))
        if _inside(span, skip) or any(not (span[1] <= a or span[0] >= b) for a, b in taken):
            continue
        sign, intpart, sep, dec, unit = m.groups()
        cands, tol, u = _candidates(sign, intpart, sep, dec, unit, text[max(0, m.start() - 30):m.start()])
        values = [c[0] for c in cands]
        base = abs(values[0])
        if dec is None and not u and base <= 10:
            continue                                                       # small counts: "3 anomalies", "2 changes"
        raw = text[span[0]:span[1]].strip()
        figs.append({'span': (span[0], span[0] + len(text[span[0]:span[1]].rstrip())), 'raw': raw, 'values': values,
                     'cands': cands, 'tol': tol, 'unit': u, 'kind': 'number'})
    figs.sort(key=lambda f: f['span'][0])
    return figs


# ------------------------------------------------------------------------------------------ context
def _norm(txt: str) -> str:
    t = txt.lower().replace('ß', 'ss')
    for a, b in (('ä', 'a'), ('ö', 'o'), ('ü', 'u'), ('é', 'e'), ('è', 'e')):
        t = t.replace(a, b)
    t = re.sub(r'strasse|str\.', 'str', t)
    t = re.sub(r'\bbhf\b|\(u\d\)', '', t)
    return re.sub(r'[^a-z0-9]+', ' ', t).strip()


class StationIndex:
    """Finds station names (in any common spelling) inside a piece of text."""
    def __init__(self, stations):
        self.names = {}
        for k in stations or []:
            n = _norm(k)
            if len(n) >= 4:
                self.names[n] = k

    def find(self, text: str) -> set:
        t = ' ' + _norm(text) + ' '
        return {k for n, k in self.names.items() if f' {n} ' in t}


UNIT_TOKENS = {'kwh': ('kwh',), 'mwh': ('mwh',), 'wh': ('wh',), 'km': ('km',), '°c': ('temp', 'tmean', 'degc'),
               'mm/h': ('rain', 'prcp', 'mm_h', 'mm/h'), 'mm': ('rain', 'prcp', 'mm')}


def _context(text: str, span):
    """Text a figure belongs to: its table row (plus the table header) or its sentence."""
    a, b = span
    ls = text.rfind('\n', 0, a) + 1
    le = text.find('\n', b); le = len(text) if le < 0 else le
    line = text[ls:le]
    if line.lstrip().startswith('|'):
        # header = first line of the contiguous table block
        start = ls
        while True:
            prev = text.rfind('\n', 0, max(start - 1, 0)) + 1
            if prev <= 0 or not text[prev:start].lstrip().startswith('|') or prev == start:
                break
            start = prev
        header = text[start:text.find('\n', start)] if text.find('\n', start) > 0 else ''
        return line, header
    s0 = max(text.rfind('. ', 0, a), text.rfind('\n', 0, a), text.rfind('; ', 0, a)) + 1
    ends = [x for x in (text.find('. ', b), text.find('\n', b), text.find('; ', b)) if x >= 0]
    return text[s0:min(ends) if ends else len(text)], ''


# ------------------------------------------------------------------------------------------ tool side
def _walk(obj, path=''):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, f'{path}.{k}' if path else str(k))
            if TIME.fullmatch(str(k)) or re.search(r'\d{2}:\d{2}', str(k)):
                yield (f'{path}.{k}' if path else str(k), 'key', str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f'{path}[{i}]')
    else:
        yield (path, 'leaf', obj)


def _parent(path):
    return re.sub(r'(\.[^.\[\]]+|\[\d+\])$', '', path)


SYN = {'passenger': ('observed', 'passenger', 'total', 'flow', 'reading', 'pax'), 'flow': ('observed', 'flow', 'total', 'reading'),
       'normal': ('normal', 'expected', 'typical', 'model'), 'expected': ('expected', 'model', 'normal', 'lambda'),
       'ceiling': ('ceiling', 'capacity'), 'capacity': ('ceiling', 'capacity'), 'attend': ('attendance', 'attendees'),
       'distance': ('km',), 'energy': ('kwh', 'mwh', 'energy'), 'rain': ('rain', 'prcp'), 'temperat': ('temp', 'tmean'),
       'correlat': ('correlation',), 'share': ('share', 'pct'), 'chance': ('chance', 'p_any', 'p_at'), 'stops': ('stops',),
       'weekday': ('weekday',), 'weekend': ('weekend',), 'station': ('station',), 'wind': ('wind', 'wspd'),
       'change': ('change', 'pct'), 'disrupt': ('disruption',), 'cut': ('cut',), 'affected': ('affected',)}


def _words(text):
    return {w[:6] for w in re.findall(r'[a-zà-ÿ]{4,}', text.lower())}


def _path_tokens(path):
    return {t[:6] for t in re.split(r'[^a-z]+', path.lower()) if len(t) >= 3}


def _kw_overlap(ctx_words, path):
    pt = _path_tokens(path)
    score = len(ctx_words & pt)
    for w in ctx_words:
        for key, toks in SYN.items():
            if w.startswith(key[:6]) and any(t[:6] in pt for t in toks):
                score += 1
    return score


def collect_tool_figures(trace, sidx: StationIndex | None = None):
    """All figures and times found in the tool calls (arguments and results). Each figure carries what it is
    about: stations and times named in its path, in the short strings of its ancestors, or in its own text."""
    nums, times = [], []
    for ci, call in enumerate(trace or []):
        try:
            data = json.loads(call.get('output', ''))
        except (json.JSONDecodeError, TypeError):
            data = {'_text': call.get('output', '')}
        for root, obj in (('args', call.get('args') or {}), ('result', data)):
            def short_strings(o):
                out = []
                for v in (o.values() if isinstance(o, dict) else o):
                    if isinstance(v, str) and len(v) <= 90:
                        out.append(v)
                    elif isinstance(v, list) and len(v) <= 12 and all(isinstance(x, str) and len(x) <= 90 for x in v):
                        out += v
                return out

            def walk(o, path, anc, par):
                if isinstance(o, dict):
                    own = ' '.join(short_strings(o))
                    anc2 = anc + ' ' + own
                    for k, v in o.items():
                        kp = f'{path}.{k}' if path else str(k)
                        for mk in re.finditer(r'(?<![\d.])(\d+(?:\.\d+)?)(?=%|pct|_min|_km|_h\b|_diverted|_day)', str(k)):
                            yield (kp, 'keynum', float(mk.group(1)), str(k), anc, own)
                        ks = re.sub(r'(\d{2})_(\d{2})', r'\1:\2', str(k))
                        for t in TIME.finditer(ks):
                            yield (kp, 'time', f'{int(t.group(1)):02d}:{t.group(2)}', str(k), anc2, own)
                        yield from walk(v, kp, anc2, own)
                elif isinstance(o, list):
                    lst = ' '.join(x for x in o if isinstance(x, str) and len(x) <= 90)
                    for i, v in enumerate(o):
                        yield from walk(v, f'{path}[{i}]', anc + ' ' + lst, lst if lst else par)
                else:
                    yield (path, 'leaf', o, None, anc, par)

            for path, kind, val, extra, anc, par in walk(obj, '', '', ''):
                full = f'{root}.{path}'
                if kind == 'time':
                    times.append((val, ci, full, extra))
                    continue
                if kind == 'keynum':
                    nums.append((val, ci, full, extra, sidx.find(anc) if sidx else set(), set(), set(), set()))
                    continue
                if isinstance(val, bool) or val is None:
                    continue
                about = f'{path} {anc}' + (f' {val}' if isinstance(val, str) else '')
                st = sidx.find(about) if sidx else set()
                tms = {f'{int(t.group(1)):02d}:{t.group(2)}' for t in TIME.finditer(re.sub(r'(\d{2})_(\d{2})', r'\1:\2', about))}
                near = re.sub(r'(\d{2})_(\d{2})', r'\1:\2', f'{path} {par}')
                slot_tms = {f'{int(t.group(1)):02d}:{t.group(2)}' for t in TIME.finditer(near)}
                lns = set(re.findall(r'\bU\d\b', f'{path} {par}'))
                if isinstance(val, (int, float)):
                    nums.append((float(val), ci, full, val, st, tms, lns, slot_tms))
                elif isinstance(val, str):
                    for t in TIME.finditer(val):
                        times.append((f'{int(t.group(1)):02d}:{t.group(2)}', ci, full, val[:200]))
                    seen = set()
                    for f in extract_figures(val):
                        if f['kind'] == 'number':
                            seen.update(f['values'][:2])
                    for x in seen:
                        nums.append((x, ci, full, val[:200], st, tms, lns, slot_tms))
    return nums, times


# ------------------------------------------------------------------------------------------ matching
def verify_answer(answer: str, trace, stations=None) -> dict:
    """Check each figure of the answer. With stations (e.g. engine.keys), a figure must come from a tool figure
    about the same station(s) and time as its sentence or table row, or share its meaning (keywords)."""
    sidx = StationIndex(stations) if stations else None
    figs = extract_figures(answer or '')
    nums, times = collect_tool_figures(trace, sidx)
    time_set = {}
    for t, ci, path, excerpt in times:
        time_set.setdefault(t, (ci, path, excerpt))
    by_parent = {}
    for rec in nums:
        by_parent.setdefault((rec[1], _parent(rec[2])), []).append(rec)
    def _block(span):
        a, b = span
        s0 = answer.rfind('\n\n', 0, a); s0 = 0 if s0 < 0 else s0
        e0 = answer.find('\n\n', b); e0 = len(answer) if e0 < 0 else e0
        return s0, e0

    def _local_vals(span):
        s0, e0 = _block(span)
        return [c for fg in figs if fg['kind'] == 'number' and s0 <= fg['span'][0] < e0 and fg['span'] != span
                for c, _ in fg['cands'][:1]]

    CUES = {'difference': r'above|below|more|fewer|less|extra|difference|higher|lower|increase|decrease|excess|additional|sav|gap|plus|minus|\+|−|-',
            'ratio': r'times|×|\bx\b|fois|ratio|factor|multiple',
            'percentage change': r'%'}

    results = []
    for f in figs:
        rec = {'span': f['span'], 'raw': f['raw'], 'status': 'not_found', 'sources': []}
        if f['kind'] == 'time':
            if f['time'] in time_set:
                ci, path, ex = time_set[f['time']]
                rec.update(status='verified', sources=[_src(trace, ci, path, ex)])
            results.append(rec)
            continue
        ctx, header = _context(answer, f['span'])
        ctx_st = sidx.find(ctx) if sidx else set()
        ctx_tm = {x['time'] for x in extract_figures(ctx) if x['kind'] == 'time'}
        ctx_words = _words(ctx + ' ' + header)
        ctx_ln = set(re.findall(r'\bU\d\b', ctx + ' ' + header))
        unit_ok = UNIT_TOKENS.get(f['unit'])
        digits = re.sub(r'\D', '', f['raw'])
        precise = (abs(f['values'][0]) >= 1000 and not digits.endswith('00')) or ('.' in f['raw'] and len(f['raw'].split('.')[-1].rstrip('%×x kmhW°C/')) >= 3)

        def score(path, raw, st, tms, lns, slot_tms):
            sc = 0.0
            if unit_ok:
                if not any(tok in (path + ' ' + str(raw)).lower() for tok in unit_ok):
                    return None
                sc += 1
            if ctx_ln and lns and not (ctx_ln & lns) and not (ctx_st and st and st & ctx_st):
                return None
            if ctx_st and st:
                if not (st & ctx_st):
                    return None
                sc += 2
            if ctx_tm and len(slot_tms) == 1 and not (slot_tms & ctx_tm):
                return None
            if ctx_tm and tms & ctx_tm:
                sc += 1
            if f['unit'].startswith('%') and re.search(r'%|pct|share|change|chance|reduction|ratio', path, re.I):
                sc += 1
            sc += min(_kw_overlap(ctx_words, path), 2)
            if precise:
                sc += 1
            return sc

        best = None
        for i, (x, ci, path, raw, st, tms, lns, slot_tms) in enumerate(nums):
            for cand, ctol in f['cands']:
                d = abs(abs(x) - abs(cand))
                if d <= ctol:
                    sc = score(path, raw, st, tms, lns, slot_tms)
                    if sc is not None and sc >= 1 and (best is None or (sc, -d) > (best[0], -best[1])):
                        best = (sc, d, i)
        if best is not None:
            x, ci, path, raw, st, tms, lns, slot_tms = nums[best[2]]
            rec.update(status='verified', sources=[_src(trace, ci, path, raw)])
        else:
            found = None
            local = _local_vals(f['span'])

            def _in_answer(x):
                return any(abs(abs(x) - abs(v)) <= max(0.006 * abs(v), 0.51) for v in local)
            cue_text = (ctx + ' ' + f['raw']).lower()
            for (ci, par), items in by_parent.items():
                items = [it for it in items[:60] if isinstance(it[3], (int, float))]
                if not items or (ctx_st and not any(it[4] & ctx_st for it in items)):
                    continue
                for i in range(len(items)):
                    for j in range(len(items)):
                        if i == j:
                            continue
                        a, pa, ra = items[i][0], items[i][2], items[i][3]
                        b, pb, rb = items[j][0], items[j][2], items[j][3]
                        derived = [('difference', a - b)]
                        if b:
                            derived += [('ratio', a / b), ('percentage change', (a / b - 1) * 100)]
                        if not (_in_answer(a) and _in_answer(b)):
                            continue
                        for name, dv in derived:
                            if not re.search(CUES[name], cue_text):
                                continue
                            if name == 'percentage change' and not f['unit'].startswith('%'):
                                continue
                            cand = f['values'][0]
                            if abs(cand) > 0 and abs(abs(dv) - abs(cand)) <= max(f['tol'], 0.004 * abs(cand)):
                                found = (name, ci, pa, ra, pb, rb)
                                break
                        if found: break
                    if found: break
                if found: break
            if found:
                name, ci, pa, ra, pb, rb = found
                rec.update(status='calculated', operation=name, sources=[_src(trace, ci, pa, ra), _src(trace, ci, pb, rb)])
        results.append(rec)
    summ = {'checked': len(results), 'verified': sum(r['status'] == 'verified' for r in results),
            'calculated': sum(r['status'] == 'calculated' for r in results),
            'not_found': sum(r['status'] == 'not_found' for r in results),
            'not_found_values': [r['raw'] for r in results if r['status'] == 'not_found']}
    return {'figures': results, 'summary': summ}


def _src(trace, ci, path, excerpt):
    call = trace[ci] if trace and ci < len(trace) else {}
    return {'call': ci + 1, 'tool': call.get('tool', '?'), 'args': call.get('args', {}), 'path': path,
            'excerpt': str(excerpt)[:220]}


# ------------------------------------------------------------------------------------------ rendering
CSS = """<style>
.vfig{background:rgba(22,163,74,.13);border-bottom:1px dotted #16a34a;border-radius:3px;padding:0 1px}
.vcal{background:rgba(37,99,235,.12);border-bottom:1px dotted #2563eb;border-radius:3px;padding:0 1px}
.vnf{background:rgba(220,38,38,.16);border-bottom:2px solid #dc2626;color:#b91c1c;border-radius:3px;padding:0 1px}
.vfig sup a,.vcal sup a{text-decoration:none;font-size:.7em;color:inherit;opacity:.75}
.vfig sup,.vcal sup{user-select:none;-webkit-user-select:none;-moz-user-select:none}
.vsrc{font-size:.8em;line-height:1.35;opacity:.9;margin:.15em 0}
.vsrc code{font-size:.95em}
</style>"""


def _source_text(r):
    parts = []
    for s in r['sources']:
        args = json.dumps(s['args'], ensure_ascii=False)
        parts.append(f"{s['tool']}({args[:120]}) → {s['path']} = {s['excerpt']}")
    head = {'verified': 'Traced to a tool result', 'calculated': f"Calculated ({r.get('operation', '')})", 'not_found': 'Not found in any tool result'}[r['status']]
    return head + (': ' + ' | '.join(parts) if parts else '')


def annotate_html(answer: str, v: dict, prefix: str = 'a') -> str:
    """Answer text with each checked figure wrapped in a colour-coded span; verified and calculated figures
    carry a superscript link to their source line (sources_html)."""
    out = answer
    for k, r in sorted(enumerate(v['figures'], 1), key=lambda x: -x[1]['span'][0]):
        a, b = r['span']
        raw = out[a:b]
        title = html.escape(_source_text(r), quote=True)
        if r['status'] == 'not_found':
            rep = f'<span class="vnf" title="{title}">{raw}</span>'
        else:
            cls = 'vfig' if r['status'] == 'verified' else 'vcal'
            rep = f'<span class="{cls}" title="{title}">{raw}<sup><a href="#{prefix}-src-{k}">{k}</a></sup></span>'
        out = out[:a] + rep + out[b:]
    return out


def sources_html(v: dict, prefix: str = 'a') -> str:
    lines = []
    for k, r in enumerate(v['figures'], 1):
        if r['status'] == 'not_found':
            txt = f"<b>{html.escape(r['raw'])}</b>: not found in any tool result"
        else:
            srcs = '; '.join(f"call {s['call']} <code>{html.escape(s['tool'])}</code> → <code>{html.escape(s['path'])}</code> = "
                             f"{html.escape(s['excerpt'][:160])}" for s in r['sources'])
            op = f" ({r['operation']})" if r['status'] == 'calculated' else ''
            txt = f"<b>{html.escape(r['raw'])}</b>{op}: {srcs}"
        lines.append(f'<div class="vsrc" id="{prefix}-src-{k}">[{k}] {txt}</div>')
    return '\n'.join(lines)


def summary_line(v: dict) -> str:
    s = v['summary']
    line = f"Figures checked: {s['checked']} · traced to a tool result {s['verified']} · calculated {s['calculated']} · not found {s['not_found']}"
    if s['not_found_values']:
        line += ' (' + ', '.join(s['not_found_values'][:8]) + ')'
    return line
