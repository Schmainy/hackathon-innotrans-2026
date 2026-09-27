"""
ubahn_reports.py - proof of value, incident reports, contingency handbook, scenario comparison, operator feedback.

    value_report(eng)                        replay the whole period: how early and how reliably alerts fire, and how much
                                             of the demand above capacity the reinforcement plans would have absorbed
    incident_report(eng, date)               post-event report of a recorded day (dict + self-contained printable HTML)
    station_contingency(eng, station)        what to do if a station closes (one handbook page, as data)
    contingency_handbook(eng)                one printable page per station (HTML)
    compare_scenarios(eng, date, a, b)       two scenarios side by side, with the stations that change the most
    log_feedback / read_feedback / feedback_summary
                                             operators rate answers and alerts; ratings go to a JSON-lines log

    python ubahn_reports.py --data data value
    python ubahn_reports.py --data data incident 2026-06-23 --out incident_2026-06-23.html
    python ubahn_reports.py --data data handbook --out contingency_handbook.html
"""
from __future__ import annotations

import argparse
import html
import json
import os
import time

import networkx as nx
import numpy as np
import pandas as pd

import src.analyses as A
import src.ops as O
from src.engine import UBahnEngine, _to_dt

MIN15 = pd.Timedelta('15min')
E = html.escape


# ================================================================================== proof of value
RULES = {
    'R1': 'two readings in a row above the normal upper bound (q95)',
    'R2': 'three readings in a row above the normal upper bound (q95)',
    'R3': 'two readings in a row above the normal 99 % bound (q99)',
    'R4': 'last hour total more than 3 standard deviations above normal',
}


def _alert_matrix(eng, idx, F, lam0, cap):
    """Boolean [slot x station] matrix of when each alert rule fires on the recorded readings."""
    q95 = O.quantile(lam0, cap, 0.95)
    q99 = O.quantile(lam0, cap, 0.99)
    a95, a99 = F > q95, F > q99
    cons = np.zeros(len(idx), bool)
    cons[1:] = (idx[1:] - idx[:-1]) == MIN15                           # slot i follows slot i-1 in the same service
    c1 = cons[:, None]
    c2 = np.zeros_like(c1); c2[2:] = c1[2:] & c1[1:-1]
    sh = lambda M, k: np.vstack([np.zeros((k, M.shape[1]), bool), M[:-k]])
    out = {'R1': a95 & sh(a95, 1) & c1, 'R2': a95 & sh(a95, 1) & sh(a95, 2) & c2, 'R3': a99 & sh(a99, 1) & c1}
    er0 = eng.expected_reading(lam0, cap)
    csF, csE, csV = (np.vstack([np.zeros((1, F.shape[1])), np.cumsum(x, axis=0)]) for x in (F, er0, er0 ** 2))
    win = lambda cs: np.vstack([np.full((3, F.shape[1]), np.nan), cs[4:] - cs[:-4]])
    z = (win(csF) - win(csE)) / np.sqrt(np.maximum(win(csV), 1))
    c4 = np.zeros(len(idx), bool)
    c4[3:] = (idx[3:] - idx[:-3]) == pd.Timedelta('45min')
    out['R4'] = np.nan_to_num(z) > 3
    out['R4'] &= c4[:, None]
    return out


def _episodes(M, gap=4):
    """First True of each run per station, separated by at least `gap` quiet slots: (slot, station) pairs."""
    res = []
    for j in range(M.shape[1]):
        on = np.where(M[:, j])[0]
        last = -10 ** 9
        for i in on:
            if i - last > gap:
                res.append((i, j))
            last = i
    return res


def value_report(eng: UBahnEngine, reinforcement: bool = True) -> dict:
    """Replay the recorded period. Early warning: for four alert rules, alerts per day, share explained by an event,
    rain or a reopening, share of events (1,000+ attendees) detected, and how far ahead of unexpected overloads they
    fire. Reinforcement: the demand above capacity (beyond a normal day) on every recorded day and the share the
    reinforcement plans would have absorbed. Takes about 20 s; cached on the engine."""
    if getattr(eng, '_value_report', None) and (eng._value_report.get('_with_reinforcement') or not reinforcement):
        return eng._value_report
    t0, t1 = eng.flows.index.min(), eng.flows.index.max()
    lam = eng.expected(t0, t1)
    lam0 = eng.expected(t0, t1, events=[])
    idx = lam.index
    F = eng.flows.reindex(idx).values
    L, L0 = lam.values, lam0.reindex(idx).values
    cap = eng.ceiling.values[None, :]
    days = sorted(set(eng.service_day(idx)))
    ndays = len(days)
    # causes that explain an alert: event crowds, rain, a reopening station
    ev_cause = (L - L0) > np.maximum(20, 0.3 * L0)
    rain = eng.weather.prcp.reindex(idx).fillna(0).values >= 0.5
    reb = np.zeros_like(ev_cause)
    pos = {t: i for i, t in enumerate(idx)}
    for c in eng._clist:
        if c['kind'] == 'station' and c['stations'] and c['end'] is not None:
            j = eng.keys.index(c['stations'][0])
            for k in range(0, 4):
                i = pos.get(c['end'].floor('15min') + k * MIN15)
                if i is not None:
                    reb[i, j] = True
    cause = ev_cause | rain[:, None] | reb
    cause_back = cause.copy()
    for k in (1, 2):
        cause_back[k:] |= cause[:-k]
    # events with 1,000+ attendees and their main stations
    evs = []
    for (vk, s0, e0), g in eng.events.dropna(subset=['venue_key']).groupby(['venue_key', 'start', 'end']):
        att = g.estimated_attendance.sum()
        if att < 1000:
            continue
        _, sh = eng.venue_shares(vk)
        main = [eng.keys.index(k) for k, v in sh.items() if v >= 0.2]
        if main:
            evs.append((vk, s0, e0, att, main))
    alerts = _alert_matrix(eng, idx, F, L0, cap)
    rules = []
    for name, M in alerts.items():
        eps = _episodes(M, gap=4)
        expl = sum(cause_back[i, j] for i, j in eps)
        det, leads_ev = 0, []
        for vk, s0, e0, att, main in evs:
            a = idx.searchsorted(e0.floor('15min'))
            b = idx.searchsorted(e0.floor('15min') + pd.Timedelta('2h'))
            hit = np.where(M[a:b][:, main].any(axis=1))[0]
            if len(hit):
                det += 1
                leads_ev.append(int(hit[0]) * 15)
        rules.append({'rule': name, 'description': RULES[name], 'alerts': len(eps), 'alerts_per_day': round(len(eps) / ndays, 1),
                      'share_explained': round(expl / max(len(eps), 1), 3),
                      'events_detected': round(det / max(len(evs), 1), 3),
                      'minutes_after_event_end_median': float(np.median(leads_ev)) if leads_ev else None,
                      'used_by_action_plan': name == 'R1'})
    for r in rules:
        r['score'] = round(r['share_explained'] * r['events_detected'], 3)
    best = max(rules, key=lambda r: r['score'])
    # planning in advance: did the action plan, made from the event calendar and the weather, flag the windows where
    # event crowds actually took a platform to its ceiling, and were the windows it flagged confirmed by the readings?
    planned = np.zeros_like(ev_cause)
    flagged, confirmed = 0, 0
    pos_day = {}
    for i, t in enumerate(idx):
        pos_day.setdefault(eng.service_day(pd.DatetimeIndex([t]))[0], []).append(i)
    ev_days = sorted(set(eng.service_day(pd.DatetimeIndex(eng.events.start))) & set(days))
    for d in ev_days:
        plan = O.action_plan(eng, str(d.date()), top=12)
        for w in plan['watch']:
            if not (w['reason'].startswith('arrivals') or w['reason'].startswith('departures')):
                continue
            j = eng.keys.index(w['station'])
            a0, b0 = w['window'].split('-')
            s0 = pd.Timestamp(f'{d.date()} {a0}')
            e0 = pd.Timestamp(f'{d.date()} {b0}')
            if e0 <= s0:
                e0 += pd.Timedelta('1D')
            ii = [i for i in pos_day.get(d, []) if s0 <= idx[i] < e0]
            if not ii:
                continue
            planned[ii, j] = True
            flagged += 1
            if (F[ii, j] >= 0.9 * cap[0, j]).any() or alerts['R1'][ii, j].any():
                confirmed += 1
    hits = ev_cause & (F >= cap)
    n_hits = int(hits.sum())
    covered = int((hits & planned).sum())
    out = {'period': f'{days[0]:%Y-%m-%d} -> {days[-1]:%Y-%m-%d}', 'service_days': ndays,
           'planning': {'event_related_readings_at_ceiling': n_hits, 'flagged_in_advance_by_the_plan': covered,
                        'share_flagged_in_advance': round(covered / n_hits, 3) if n_hits else None,
                        'event_windows_flagged': flagged, 'share_confirmed_by_readings': round(confirmed / flagged, 3) if flagged else None,
                        'definitions': {'event-related reading at ceiling': 'a reading at its ceiling while an event crowd adds at least 20 passengers '
                                                                             'and 30 % to the normal level',
                                        'flagged in advance': "inside a window of that day's action plan (made from the event calendar and weather)",
                                        'confirmed': 'a reading at 90 % of the ceiling or a surge alert inside the flagged window'}},
           'early_warning': {'rules': rules, 'recommended_rule': best['rule'], 'events_1000_plus': len(evs),
                             'action_plan_rule': 'the action plan applies R1 only inside its watch windows, where it is most sensitive; network-wide, the recommended rule is the better trade-off',
                             'note': 'single-station alerts are noisy in this data: most alerts are not tied to a known cause, because readings '
                                     'fluctuate far more than a normal day suggests; use them as prompts to look at a station, and rely on the '
                                     'plan made in advance for known events',
                             'definitions': {'explained': 'an event crowd, rain of 0.5 mm/h or more, or a station reopening within the 30 minutes before the alert',
                                             'event detected': 'an alert at one of its main stations within 2 hours after it ends'}}}
    if reinforcement:
        rows = []
        for d in days:
            rp = O.reinforcement_plan(eng, str(d.date()), top=20)
            ext = sum(w['extra_total'] for w in rp['windows'])
            if ext <= 0:
                continue
            by = {}
            for w in rp['windows']:
                for m, v in w['absorbed_by_mode'].items():
                    by[m] = by.get(m, 0) + v
            res_ = sum(w['residual_total'] for w in rp['windows'])
            reasons = sorted(set(w['reason'].split(' (')[0] for w in rp['windows']))
            rows.append({'date': f'{d:%Y-%m-%d}', 'weekday': d.day_name()[:3], 'extra_above_capacity': ext,
                         'absorbed': min(sum(by.values()), ext), 'residual': res_, 'by_mode': by,
                         'main_reasons': reasons[:3], 'worst_window': max(rp['windows'], key=lambda w: w['extra_total'])['station'] + ' ' +
                         max(rp['windows'], key=lambda w: w['extra_total'])['window']})
        tot = sum(r['extra_above_capacity'] for r in rows)
        absb = sum(r['absorbed'] for r in rows)
        by_mode = {}
        for r in rows:
            for m, v in r['by_mode'].items():
                by_mode[m] = by_mode.get(m, 0) + v
        top_days = sorted(rows, key=lambda r: -r['extra_above_capacity'])[:10]
        out['reinforcement'] = {'days_with_extra_demand': len(rows), 'extra_above_capacity_total': round(tot),
                                'absorbed_total': round(absb), 'share_absorbed': round(absb / tot, 3) if tot else None,
                                'absorbed_by_mode': {k: round(v) for k, v in sorted(by_mode.items(), key=lambda x: -x[1])},
                                'residual_total': round(sum(r['residual'] for r in rows)),
                                'top_days': top_days,
                                'basis': 'demand above capacity beyond a normal day at the same temperature; other modes rest on the planning assumptions of ubahn_reinforcement_plan'}
    b = best
    pl = out['planning']
    heads = []
    if pl['share_flagged_in_advance'] is not None:
        heads.append(f"Planning in advance: the action plan, made from the event calendar and the weather, had flagged "
                     f"{pl['share_flagged_in_advance']:.0%} of the {pl['event_related_readings_at_ceiling']:,} event-related readings at a platform's ceiling; "
                     f"{pl['share_confirmed_by_readings']:.0%} of the {pl['event_windows_flagged']} event windows it flagged were confirmed by the readings.")
    heads.append(f"Live alerts: the best rule ({b['rule']}, {b['description']}) flags {b['events_detected']:.0%} of the {len(evs)} events with 1,000+ attendees"
                 + (f", a median {b['minutes_after_event_end_median']:.0f} minutes after they end" if b['minutes_after_event_end_median'] is not None else '')
                 + f"; it raises {b['alerts_per_day']} alerts a day on the network, {b['share_explained']:.0%} of them tied to an event, rain or a reopening.")
    if reinforcement and out['reinforcement']['share_absorbed'] is not None:
        rr = out['reinforcement']
        heads.append(f"Reinforcement: over {rr['days_with_extra_demand']} days with extra demand, the plans would have absorbed, under their planning assumptions, "
                     f"{rr['share_absorbed']:.0%} of the {rr['extra_above_capacity_total']:,} passengers above capacity beyond a normal day "
                     f"({', '.join(f'{k} {v:,}' for k, v in list(rr['absorbed_by_mode'].items())[:4])}).")
        w = rr['top_days'][0]
        heads.append(f"Most demanding day: {w['date']} ({', '.join(w['main_reasons'])}): {w['extra_above_capacity']:,} passengers above capacity, "
                     f"{w['absorbed'] / max(w['extra_above_capacity'], 1):.0%} absorbed, {w['residual']:,} left to meter.")
    out['headlines'] = heads
    out = A._py(out)
    out['_with_reinforcement'] = reinforcement
    eng._value_report = out
    return out


# ================================================================================== incident report
CSS = """<style>
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#111827;line-height:1.45}
h1{font-size:1.6em;margin-bottom:.1em} h2{font-size:1.2em;margin-top:1.6em;border-bottom:2px solid #e5e7eb;padding-bottom:.2em}
h3{font-size:1.02em;margin:1em 0 .3em} .meta{color:#6b7280;font-size:.85em}
table{border-collapse:collapse;width:100%;margin:.4em 0 .8em;font-size:.88em} th,td{border:1px solid #e5e7eb;padding:4px 6px;text-align:left;vertical-align:top}
th{background:#f3f4f6} .box{background:#f9fafb;border-left:4px solid #2563eb;padding:8px 12px;margin:.6em 0}
.lesson{border-left-color:#16a34a} .warn{border-left-color:#dc2626} ul{margin:.3em 0 .6em 1.2em;padding:0}
.msg{font-size:.86em;background:#fff;border:1px solid #e5e7eb;border-radius:4px;padding:6px 8px;margin:4px 0}
.toc{columns:3;font-size:.85em} .toc a{text-decoration:none} .page{page-break-before:always;break-before:page}
@media print{body{margin:0;max-width:none} h2{page-break-after:avoid}}
</style>"""


def _table(rows, cols, labels=None):
    if not rows:
        return '<p class="meta">None.</p>'
    head = ''.join(f'<th>{E(str(l))}</th>' for l in (labels or cols))
    body = ''.join('<tr>' + ''.join(f'<td>{E(str(r.get(c, "")) if r.get(c) is not None else "—")}</td>' for c in cols) + '</tr>' for r in rows)
    return f'<table><tr>{head}</tr>{body}</table>'


def _ul(items, cls=None):
    if not items:
        return '<p class="meta">None.</p>'
    return (f'<div class="box {cls}">' if cls else '') + '<ul>' + ''.join(f'<li>{E(str(i))}</li>' for i in items) + '</ul>' + ('</div>' if cls else '')


def incident_report(eng: UBahnEngine, date: str) -> dict:
    """Post-event report of a recorded service day, built only from the tools: situation, events and closures with
    their recorded impact, anomalies, when the alerts would have fired, reinforcement and staff that the plans would
    have recommended, passenger messages and lessons learned. Returns the content and a printable HTML page."""
    day = pd.Timestamp(date).normalize()
    S, En = day + pd.Timedelta('5h'), day + pd.Timedelta('1D 45min')
    if not (S >= eng.flows.index.min() and En <= eng.flows.index.max()):
        return {'error': f'{day:%Y-%m-%d} is not a fully recorded service day '
                         f'(data {eng.flows.index.min():%Y-%m-%d} -> {eng.flows.index.max():%Y-%m-%d})'}
    brief = O.daily_brief(eng, date, staff=10)
    plan = O.action_plan(eng, date, top=8)
    rp = O.reinforcement_plan(eng, date, top=8)
    an = A.anomaly_scan(eng, date)
    events_out = []
    for e in brief['events']:
        try:
            er = A.event_report(eng, e['name'][:40], date)
            events_out.append({'name': e['name'], 'venue': e['venue'], 'attendance': e['attendance'],
                               'facts': er.get('facts_from_recorded_data', [])[1:5],
                               'departure_extra': er.get('departure_window', {}).get('extra_passengers'),
                               'share': er.get('departure_window', {}).get('extra_as_share_of_attendance')})
        except Exception as ex:
            events_out.append({'name': e['name'], 'venue': e['venue'], 'attendance': e['attendance'], 'facts': [f'not analysed: {ex}']})
    closures_out = []
    for c in brief['closures']:
        cc = next((x for x in eng._clist if x['description'] == c['description'] and x['start'] == _to_dt(c['start'])), None)
        facts = []
        if cc is not None and cc['kind'] == 'line':
            facts.append('Line suspensions leave no measurable trace in the flows: passengers wait or use other modes.')
        elif cc is not None and cc['kind'] == 'station':
            k = cc['stations'][0]
            a, b = cc['start'].floor('15min'), cc['end']
            lost = float(eng.expected_reading(eng.expected(a, b - MIN15, [k], closures=[]).values, eng.ceiling[[k]].values[None, :]).sum())
            after = eng.flows.loc[b.floor('15min'):b.floor('15min') + pd.Timedelta('30min'), k]
            facts.append(f'{k} recorded zero passengers while closed (about {lost:,.0f} expected); first readings after reopening: '
                         + ', '.join(f'{int(v)}' for v in after.values) + '.')
        msgs = O.passenger_messages(eng, closure=c['description'], start=str(c['start']), end=str(c['end']))
        closures_out.append({'description': c['description'], 'start': c['start'], 'end': c['end'], 'facts': facts,
                             'message_en': msgs.get('messages', {}).get('en', {}).get('announcement')})
    replay = []
    lessons = []
    for w in plan['watch']:
        fired = w.get('replay_on_recorded_data', {}).get('would_have_fired_at', {})
        replay.append({'station': w['station'], 'window': w['window'], 'reason': w['reason'],
                       'surge': fired.get('surge') or '—', 'above_plan': fired.get('above_plan') or '—',
                       'near_capacity': fired.get('near_capacity') or '—'})
        if fired.get('near_capacity'):
            if fired.get('surge') and fired['surge'] <= fired['near_capacity']:
                gap = (pd.Timestamp(f"2000-01-01 {fired['near_capacity']}") - pd.Timestamp(f"2000-01-01 {fired['surge']}")).seconds // 60
                lessons.append(f"{w['station']}: the surge alert fired at {fired['surge']}, {gap} minutes before the platform reached 90 % of its capacity ({fired['near_capacity']}).")
            else:
                lessons.append(f"{w['station']}: the platform reached 90 % of its capacity at {fired['near_capacity']} without an earlier surge alert; "
                               f"the planned 'prepare' step ({w['reason']}) is what covers this case.")
        if fired.get('above_plan'):
            lessons.append(f"{w['station']}: demand went above the plan at {fired['above_plan']}; check the event attendance or unannounced causes.")
        if fired.get('surge') and not fired.get('near_capacity'):
            lessons.append(f"{w['station']}: a crowd bigger than a normal day was detected at {fired['surge']} ({w['reason']}), without reaching capacity.")
    for e in events_out:
        if e.get('share') is not None and e.get('departure_extra') is not None and e['departure_extra'] > 0:
            lessons.append(f"After {e['name']}, about {e['share']:.0%} of the {int(e['attendance']):,} attendees took the U-Bahn within two hours "
                           f"(+{int(e['departure_extra']):,} passengers).")
    for c in closures_out:
        lessons += [f"{c['description']} {f}" for f in c['facts'][:1]]
    ext = sum(w['extra_total'] for w in rp['windows'])
    absb = sum(sum(w['absorbed_by_mode'].values()) for w in rp['windows'])
    if ext > 0:
        lessons.append(f"Reinforcement would have absorbed about {min(absb, ext) / ext:.0%} of the {ext:,} passengers above capacity beyond a normal day.")
    rein_rows = [{'station': w['station'], 'window': w['window'], 'reason': w['reason'],
                  'measures': ' · '.join(f"{m['mode']}: {m['action']}" for m in w['measures'])} for w in rp['windows']]
    anomalies = [f"{f['type']}: {f['when']}, {f['most_likely_cause']} ({f['evidence']})" for f in an.get('ranked_findings', [])]
    net_obs = plan.get('network_recorded_passengers')
    content = {'date': day, 'weekday': day.day_name(), 'headlines': brief['headlines'], 'events': events_out, 'closures': closures_out,
               'anomalies': anomalies, 'alerts_replay': replay, 'reinforcement': rein_rows, 'reinforcement_summary': rp['summary'],
               'staff_plan': brief['staff_plan'], 'lessons_learned': lessons,
               'network': {'recorded': net_obs, 'expected': plan['network_expected_passengers'], 'normal': plan['network_normal_passengers']}}
    h = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Incident report {day:%Y-%m-%d}</title>{CSS}</head><body>',
         f'<h1>Incident report · {day:%A %d %B %Y}</h1>',
         f'<p class="meta">Generated {time.strftime("%Y-%m-%d %H:%M")} from the recorded data ({eng.flows.index.min():%d %b} – {eng.flows.index.max():%d %b %Y}). '
         f'Every figure comes from the analysis tools; nothing is written by a language model.</p>',
         '<h2>Situation</h2>', _ul(brief['headlines']),
         '<h2>Lessons learned</h2>', _ul(lessons, 'lesson'),
         '<h2>Events and their recorded impact</h2>']
    for e in events_out:
        h.append(f"<h3>{E(e['name'])} · {E(str(e['venue']))} · {int(e['attendance']):,} attendees</h3>" + _ul(e['facts']))
    if not events_out:
        h.append('<p class="meta">No event.</p>')
    h.append('<h2>Closures</h2>')
    for c in closures_out:
        h.append(f"<h3>{E(c['description'])} ({E(str(c['start'])[11:16])}–{E(str(c['end'])[11:16])})</h3>" + _ul(c['facts'] or ['—'])
                 + (f'<div class="msg"><b>Passenger announcement:</b> {E(c["message_en"])}</div>' if c.get('message_en') else ''))
    if not closures_out:
        h.append('<p class="meta">No closure.</p>')
    h += ['<h2>Anomalies</h2>', _ul(anomalies),
          '<h2>Alerts replayed on the recorded readings</h2>',
          '<p class="meta">When each trigger of the action plan would have fired. Surge: two readings above the normal level\'s upper bound; '
          'above plan: two readings above the forecast\'s upper bound; near capacity: a reading at 90 % of the ceiling.</p>',
          _table(replay, ['station', 'window', 'reason', 'surge', 'above_plan', 'near_capacity'],
                 ['Station', 'Window', 'Reason', 'Surge', 'Above plan', 'Near capacity']),
          '<h2>Reinforcement the plan would have recommended</h2>', _ul(rp['summary']),
          _table(rein_rows, ['station', 'window', 'reason', 'measures'], ['Station', 'Window', 'Reason', 'Measures']),
          f'<p class="meta">{E(rp["caution"])}.</p>',
          '<h2>Additional staff</h2>',
          _table(brief['staff_plan'], ['station', 'from', 'to', 'staff', 'reason'], ['Station', 'From', 'To', 'Staff', 'Reason']),
          '</body></html>']
    return A._py({**content, 'html': '\n'.join(h)})


# ================================================================================== contingency handbook
def station_contingency(eng: UBahnEngine, station: str) -> dict:
    """What to do if a station closes: passengers affected, parts of the network cut off if its tracks are out,
    walking and S-Bahn alternatives, reinforcement and ready passenger messages (end time left as [HH:MM])."""
    k = eng.resolve_station(station)
    G = eng.graph
    daily, wd, we = A._daily(eng)
    H = G.copy(); H.remove_node(k)
    comps = sorted(nx.connected_components(H), key=len, reverse=True)
    pieces = []
    for c in comps[1:]:
        c = list(c)
        far = max(c, key=lambda x: nx.shortest_path_length(G, k, x))
        pieces.append({'stations': len(c), 'towards': far, 'lines': sorted(set(sum([eng.stations.at[x, 'lines'] for x in c], []))),
                       'weekday_passengers': round(float(wd[c].sum()))})
    lines = O._lines_of(eng, k)
    lat, lon = eng.stations.at[k, 'latitude'], eng.stations.at[k, 'longitude']
    adjacent = [n for n in G.neighbors(k)]
    walk = [{'station': s2, 'km': d, 'lines': O._lines_of(eng, s2)} for s2, d in eng.stations_near(lat, lon, 1.2) if s2 != k][:5]
    other = [w for w in walk if set(w['lines']) - set(lines)]
    sb = O._sbahn_options(eng, k, 1.0)
    measures = ['keep trains running through the station if the tracks are usable (then only its own passengers are affected)',
                f"otherwise run a replacement bus between {' and '.join(adjacent[:2])}" if len(adjacent) >= 2 else 'otherwise run a replacement bus to the nearest station']
    if other:
        measures.append(f"reinforce {'/'.join(other[0]['lines'])} at {other[0]['station']} ({other[0]['km']} km on foot) and guide passengers there")
    if sb:
        measures.append(f"guide passengers to the S-Bahn at {sb[0][0]}" + (f' ({sb[0][1]} km)' if sb[0][1] else ''))
    if pieces:
        measures.append('shuttle trains on each cut-off branch: ' + '; '.join(f"towards {p['towards']} ({p['stations']} stations)" for p in pieces))
    msgs = O.passenger_messages(eng, closure=f'Station {k} closed', start='2026-01-01 10:00', end='2026-01-01 12:00', until_label='[HH:MM]')
    si = A.station_info(eng, k)
    return A._py({'station': k, 'lines': lines, 's_bahn_interchange': bool(A._sbahn(eng, k)), 'ceiling_15min': int(eng.ceiling[k]),
                  'weekday_passengers': round(float(wd[k])), 'weekend_passengers': round(float(we[k])),
                  'if_tracks_out': {'stations_cut_off': sum(p['stations'] for p in pieces), 'pieces': pieces,
                                    'passengers_affected_weekday': round(float(wd[k] + sum(p['weekday_passengers'] for p in pieces)))},
                  'adjacent_stations': adjacent, 'walking_alternatives': walk, 's_bahn': [{'station': s_, 'km': d_, 'kind': t_} for s_, d_, t_ in sb],
                  'measures': measures, 'event_venues_nearby': si.get('event_venues_nearby', [])[:3],
                  'past_closures': si.get('closures_in_data', [])[:3], 'passenger_messages': msgs.get('messages', {})})


def contingency_handbook(eng: UBahnEngine, stations: list | None = None) -> dict:
    """One printable page per station (HTML), with an index by line."""
    keys = [eng.resolve_station(s) for s in stations] if stations else sorted(eng.keys, key=lambda k: (O._lines_of(eng, k)[0], k))
    pages = [station_contingency(eng, k) for k in keys]
    anchor = {p['station']: f"st-{i}" for i, p in enumerate(pages)}
    by_line = {}
    for p in pages:
        for L in p['lines']:
            by_line.setdefault(L, []).append(p['station'])
    h = [f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>U-Bahn contingency handbook</title>{CSS}</head><body>',
         '<h1>U-Bahn contingency handbook</h1>',
         f'<p class="meta">One page per station: what to do if it closes. Generated {time.strftime("%Y-%m-%d %H:%M")} from the data '
         f'({eng.flows.index.min():%d %b} – {eng.flows.index.max():%d %b %Y}). Passenger figures are typical weekday or weekend days. '
         'Messages leave the end time as [HH:MM]. S-Bahn, bus and walking capacities are not in the data: check them locally.</p>',
         '<h2>Index</h2>']
    for L in sorted(by_line):
        h.append(f'<h3>{E(L)}</h3><div class="toc">' + ' · '.join(f'<a href="#{anchor[s]}">{E(s)}</a>' for s in by_line[L]) + '</div>')
    for p in pages:
        t = p['if_tracks_out']
        h.append(f'<div class="page" id="{anchor[p["station"]]}"><h2>{E(p["station"])} · {E("/".join(p["lines"]))}'
                 f'{" · S-Bahn interchange" if p["s_bahn_interchange"] else ""}</h2>')
        h.append(f'<p>Typical day: <b>{p["weekday_passengers"]:,}</b> passengers on weekdays, {p["weekend_passengers"]:,} on weekends; '
                 f'ceiling (highest recorded flow) {p["ceiling_15min"]} per 15 minutes.</p>')
        h.append('<h3>If the station closes</h3>' + _ul([
            f'Trains pass through: {p["weekday_passengers"]:,} weekday passengers affected.',
            (f'Tracks out: {t["stations_cut_off"]} stations cut off, {t["passengers_affected_weekday"]:,} weekday passengers affected ('
             + '; '.join(f"{x['stations']} stations towards {x['towards']}, {x['weekday_passengers']:,} passengers" for x in t['pieces']) + ').')
            if t['pieces'] else 'Tracks out: no station is cut off (the network stays connected).']))
        h.append('<h3>Measures</h3>' + _ul(p['measures'], 'lesson'))
        h.append('<h3>Alternatives</h3>' + _table(p['walking_alternatives'], ['station', 'km', 'lines'], ['Station', 'Walk (km)', 'Lines']))
        if p['s_bahn']:
            h.append('<p>S-Bahn: ' + E(', '.join(f"{x['station']}" + (f" ({x['km']} km)" if x['km'] else '') for x in p['s_bahn'])) + '</p>')
        if p['event_venues_nearby']:
            h.append('<p class="meta">Event venues nearby: ' + E(', '.join(str(v.get('venue', v)) for v in p['event_venues_nearby'])) + '</p>')
        pm = p['passenger_messages']
        if pm:
            h.append('<h3>Passenger messages</h3>' + ''.join(
                f'<div class="msg"><b>{lab}:</b> {E(pm[lang]["announcement"])}<br><i>{E(pm[lang]["display"])}</i></div>'
                for lang, lab in (('de', 'Deutsch'), ('en', 'English'), ('fr', 'Français')) if lang in pm))
        h.append('</div>')
    h.append('</body></html>')
    return {'stations': len(pages), 'html': '\n'.join(h), 'lines': sorted(by_line)}


# ================================================================================== scenario comparison
def _scen_tables(eng, day, sc):
    w = None
    if sc.get('tmean') is not None or sc.get('prcp') is not None:
        w = {'tmean': float(sc.get('tmean', 16.0) if sc.get('tmean') is not None else 16.0), 'prcp': float(sc.get('prcp') or 0.0)}
    S, En, in_data, w2, evs, cls, notes = O._scenario(eng, day, w, sc.get('extra_events') or None, sc.get('extra_closures') or None)
    lam = eng.expected(S, En, weather=w2, events=evs, closures=cls)
    cap = eng.ceiling.values[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(lam.values > 0, np.exp(-cap / np.maximum(lam.values, 1e-9)), 0)
    over = lam.values * p
    er = eng.expected_reading(lam.values, cap)
    return lam.index, p, over, er, notes


def compare_scenarios(eng: UBahnEngine, date: str, scenario_a: dict | None = None, scenario_b: dict | None = None, top: int = 10):
    """Two scenarios of the same service day side by side (each: label, tmean, prcp, extra_events, extra_closures;
    an empty scenario is the day as recorded or a dry 16 degC day): network totals, stations likely to reach their
    ceiling, passengers above capacity, the stations that change the most, and each scenario's reinforcement."""
    day = pd.Timestamp(date).normalize()
    a, b = dict(scenario_a or {}), dict(scenario_b or {})
    a.setdefault('label', 'A'); b.setdefault('label', 'B')
    res = {}
    for sc in (a, b):
        idx, p, over, er, notes = _scen_tables(eng, day, sc)
        pmax = p.max(axis=0)
        res[sc['label']] = {'idx': idx, 'p': p, 'over': over, 'er': er, 'notes': notes,
                            'summary': {'network_expected_passengers': round(float(er.sum())),
                                        'passengers_above_capacity': round(float(over.sum())),
                                        'stations_likely_at_ceiling': int((pmax >= 0.5).sum()),
                                        'highest_risk': [{'station': eng.keys[j], 'chance_at_ceiling': round(float(pmax[j]), 2),
                                                          'at': idx[int(np.argmax(p[:, j]))].strftime('%H:%M')}
                                                         for j in np.argsort(-pmax)[:5]]}}
    ra, rb = res[a['label']], res[b['label']]
    d_over = rb['over'].sum(axis=0) - ra['over'].sum(axis=0)
    d_pax = rb['er'].sum(axis=0) - ra['er'].sum(axis=0)
    d_p = rb['p'].max(axis=0) - ra['p'].max(axis=0)
    order = np.argsort(-(np.abs(d_over) + 0.02 * np.abs(d_pax)))[:top]
    changes = [{'station': eng.keys[j], 'lines': O._lines_of(eng, eng.keys[j]),
                'passengers_change': round(float(d_pax[j])), 'above_capacity_change': round(float(d_over[j])),
                'max_chance_at_ceiling': [round(float(ra['p'][:, j].max()), 2), round(float(rb['p'][:, j].max()), 2)]}
               for j in order if abs(d_pax[j]) >= 1 or abs(d_over[j]) >= 1]
    rein = {}
    for sc in (a, b):
        w = None
        if sc.get('tmean') is not None or sc.get('prcp') is not None:
            w = {'tmean': float(sc.get('tmean') if sc.get('tmean') is not None else 16.0), 'prcp': float(sc.get('prcp') or 0.0)}
        rein[sc['label']] = O.reinforcement_plan(eng, date, w, sc.get('extra_events') or None, sc.get('extra_closures') or None, top=6)['summary']
    sa, sb = ra['summary'], rb['summary']
    heads = [f"{b['label']} vs {a['label']}: {sb['network_expected_passengers'] - sa['network_expected_passengers']:+,} passengers on the network, "
             f"{sb['passengers_above_capacity'] - sa['passengers_above_capacity']:+,} passengers above capacity, "
             f"{sb['stations_likely_at_ceiling'] - sa['stations_likely_at_ceiling']:+d} stations likely to reach their ceiling."]
    if changes:
        c0 = changes[0]
        heads.append(f"Biggest change: {c0['station']} ({c0['passengers_change']:+,} passengers, chance of reaching the ceiling "
                     f"{c0['max_chance_at_ceiling'][0]:.0%} -> {c0['max_chance_at_ceiling'][1]:.0%}).")
    return A._py({'date': day, 'scenarios': {a['label']: {k: v for k, v in a.items()}, b['label']: {k: v for k, v in b.items()}},
                  'summary': {a['label']: sa, b['label']: sb}, 'notes': {a['label']: ra['notes'], b['label']: rb['notes']},
                  'biggest_changes': changes, 'reinforcement': rein, 'headlines': heads})


# ================================================================================== operator feedback
def log_feedback(path: str, record: dict) -> bool:
    """Append one rating (kind: answer | alert, rating: up | down, plus context) to a JSON-lines log. Never raises."""
    try:
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        rec = {'time': time.strftime('%Y-%m-%d %H:%M:%S'), **record}
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + '\n')
        return True
    except Exception:
        return False


def read_feedback(path: str) -> list:
    out = []
    if path and os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def feedback_summary(entries: list) -> dict:
    s = {}
    for kind in ('answer', 'alert'):
        e = [x for x in entries if x.get('kind') == kind]
        up = sum(x.get('rating') == 'up' for x in e)
        s[kind] = {'ratings': len(e), 'up': up, 'down': len(e) - up, 'share_up': round(up / len(e), 3) if e else None}
    s['comments'] = [{'time': x['time'], 'kind': x.get('kind'), 'rating': x.get('rating'), 'about': x.get('about', ''), 'comment': x['comment']}
                     for x in entries if x.get('comment')][-10:]
    return s


# ================================================================================== command line
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True)
    ap.add_argument('what', choices=['value', 'incident', 'handbook'])
    ap.add_argument('date', nargs='?', help='service day for an incident report, e.g. 2026-06-23')
    ap.add_argument('--out', help='HTML file to write (incident, handbook)')
    a = ap.parse_args()
    eng = UBahnEngine(a.data)
    if a.what == 'value':
        r = value_report(eng)
        print('\n'.join(r['headlines']))
        print(pd.DataFrame(r['early_warning']['rules']).drop(columns=['description']).to_string(index=False))
        return
    if a.what == 'incident':
        if not a.date:
            ap.error('give the date of the day to report on')
        r = incident_report(eng, a.date)
        if 'error' in r:
            raise SystemExit(r['error'])
    else:
        r = contingency_handbook(eng)
    out = a.out or (f'incident_{a.date}.html' if a.what == 'incident' else 'contingency_handbook.html')
    with open(out, 'w', encoding='utf-8') as f:
        f.write(r['html'])
    print(f'written: {out} ({len(r["html"]) // 1024} KB)')


if __name__ == '__main__':
    main()
