"""
ubahn_ops.py - operational features built on UBahnEngine.

    forecast          expected level per station and 15-min slot with calibrated plausible ranges (80 % / 90 %)
    action_plan       conditional plan for a day or scenario: what to watch, thresholds, "if ... then ..." triggers,
                      and, for a day in the data, when each trigger would really have fired
    staff_plan        where to put N staff members, and when, to cover the most passengers above capacity
    daily_brief       proactive morning brief: weather, events, closures, risks, staff plan, key triggers
    snapshot          what the network looked like at one moment ("what would we have seen at 22:45?")
    engine_state / learning_report
                      what the model learned from newly injected data (new venues, new days, changed levels)

Ranges: a 15-minute reading is the expected level times a random exponential factor, capped at the station
ceiling. The 80 % range is [q10, q90] and the 90 % range [q5, q95] of that distribution; on the whole training
summer they contain 82 % and 92 % of the real readings (band_coverage).
"""
from __future__ import annotations

import json
import re
import os

import numpy as np
import pandas as pd

import src.analyses as A
from src.engine import TOD_ORDER, UBahnEngine, _to_dt

MIN15 = pd.Timedelta('15min')
DEFAULT_SCENARIO_WEATHER = {'tmean': 16.0, 'prcp': 0.0}


# ------------------------------------------------------------------------------------------ helpers
def quantile(lam, cap, p):
    """p-quantile of a reading with expected level lam and ceiling cap (exponential noise, floored, capped)."""
    lam = np.asarray(lam, float)
    q = np.floor(-lam * np.log(1 - p))
    return np.where(lam > 0, np.minimum(q, cap), 0)


def band_coverage(eng: UBahnEngine) -> dict:
    """Share of all recorded readings (expected level > 5) inside the 80 % and 90 % ranges. Cached."""
    if getattr(eng, '_band_cov', None):
        return eng._band_cov
    t0, t1 = eng.flows.index.min(), eng.flows.index.max()
    lam = eng.expected(t0, t1)
    O = eng.flows.reindex(lam.index).values
    L = lam.values
    cap = eng.ceiling.values[None, :]
    m = L > 5
    out = {}
    for name, (lo, hi) in (('80', (0.10, 0.90)), ('90', (0.05, 0.95))):
        inside = (O >= quantile(L, cap, lo)) & (O <= quantile(L, cap, hi))
        out[name] = round(float(inside[m].mean()), 3)
    above90 = (O > quantile(L, cap, 0.95))[m].mean()
    out['share_above_90_upper'] = round(float(above90), 3)
    eng._band_cov = out
    return out


def _scenario(eng, day, weather, extra_events, extra_closures):
    """Weather, events and closures for a service day: recorded data when available, plus scenario additions."""
    S, E = day + pd.Timedelta('5h'), day + pd.Timedelta('1D 45min')
    in_data = S >= eng.flows.index.min() and E <= eng.flows.index.max()
    has_weather = np.isfinite(eng._daily_temp(eng.slots(S, E))).all()
    notes = []
    if weather is None and not has_weather:
        weather = dict(DEFAULT_SCENARIO_WEATHER)
        notes.append(f"no recorded weather for this day: assumed {weather['tmean']} degC daily mean, dry")
    evs = eng._data_events(S, E) + [dict(e) for e in (extra_events or [])]
    cls = list(eng._clist) + [eng.parse_closure(c.get('description', ''), c.get('start'), c.get('end'))
                              if isinstance(c, dict) else eng.parse_closure(*c) for c in (extra_closures or [])]
    return S, E, in_data, weather, evs, cls, notes


def _rain(eng, weather, t0, t1):
    if weather is not None:
        return float(weather.get('prcp', 0) or 0)
    w = eng.weather.prcp.loc[t0:t1]
    return float(w.max()) if len(w) else 0.0


def _lines_of(eng, st):
    return list(eng.stations.at[st, 'lines'])


# ------------------------------------------------------------------------------------------ forecast
def forecast(eng: UBahnEngine, start: str, end: str, stations: list | None = None, line: str | None = None,
             weather: dict | None = None, extra_events: list | None = None, extra_closures: list | None = None):
    """Expected reading per station and slot with plausible ranges, the normal level (no events) and, when the
    window is in the data, the recorded reading and whether it fell inside the 80 % range."""
    keys, scope = A._scope(eng, stations, line)
    s0, e0 = _to_dt(start), _to_dt(end)
    day = eng.service_day(pd.DatetimeIndex([s0]))[0]
    _, _, _, weather, evs, cls, notes = _scenario(eng, day, weather, extra_events, extra_closures)
    lam = eng.expected(s0, e0, keys, weather=weather, events=evs, closures=cls)
    lam0 = eng.expected(s0, e0, keys, weather=weather, events=[], closures=cls)
    cap = eng.ceiling[keys].values[None, :]
    er = eng.expected_reading(lam.values, cap)
    rows = []
    obs = eng.flows.reindex(lam.index)[keys] if lam.index.max() <= eng.flows.index.max() else None
    for j, k in enumerate(keys[:12]):
        for i, t in enumerate(lam.index):
            r = {'station': k, 'slot': t.strftime('%Y-%m-%d %H:%M'), 'expected': round(float(er[i, j])),
                 'range_80': [int(quantile(lam.values[i, j], cap[0, j], 0.10)), int(quantile(lam.values[i, j], cap[0, j], 0.90))],
                 'range_90': [int(quantile(lam.values[i, j], cap[0, j], 0.05)), int(quantile(lam.values[i, j], cap[0, j], 0.95))],
                 'normal_upper_90': int(quantile(lam0.values[i, j], cap[0, j], 0.95)),
                 'chance_at_ceiling': round(float(np.exp(-cap[0, j] / lam.values[i, j])) if lam.values[i, j] > 0 else 0.0, 3),
                 'ceiling': int(cap[0, j])}
            if obs is not None and np.isfinite(obs.iat[i, j]):
                r['recorded'] = int(obs.iat[i, j])
                r['recorded_inside_80'] = bool(r['range_80'][0] <= r['recorded'] <= r['range_80'][1])
            rows.append(r)
    cov = band_coverage(eng)
    return A._py({'scope': scope, 'stations_shown': keys[:12], 'window': [s0, e0], 'notes': notes,
                  'calibration': f"on the training summer the 80 % range held {cov['80']:.0%} of real readings and the 90 % range {cov['90']:.0%}",
                  'slots': rows})


# ------------------------------------------------------------------------------------------ action plan
def _risk_tables(eng, S, E, weather, evs, cls):
    lam = eng.expected(S, E, weather=weather, events=evs, closures=cls)
    lam0 = eng.expected(S, E, weather=weather, events=[], closures=cls)
    cap = eng.ceiling.values[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(lam.values > 0, np.exp(-cap / np.maximum(lam.values, 1e-9)), 0)
    over = lam.values * p
    evload = np.maximum(lam.values - lam0.values, 0)
    return lam, lam0, p, over, evload


def _event_reason(eng, evs, st, t):
    for e in evs:
        _, shares = eng.venue_shares(e.get('venue') or e.get('address'))
        if st not in shares:
            continue
        s0, e0 = _to_dt(e['start']), _to_dt(e['end'])
        if s0 - pd.Timedelta('90min') <= t <= s0 + MIN15:
            return f"arrivals for {e.get('name', 'event')} ({int(e['attendance'])} attendees, {s0:%H:%M})"
        if e0 <= t <= e0 + pd.Timedelta('2h'):
            return f"departures after {e.get('name', 'event')} ({int(e['attendance'])} attendees, ends {e0:%H:%M})"
    return None


def action_plan(eng: UBahnEngine, date: str, weather: dict | None = None, extra_events: list | None = None,
                extra_closures: list | None = None, top: int = 8):
    """Conditional plan for one service day: the station windows to watch, calibrated thresholds, triggers and
    actions, each with its indicator. For a day in the data, every trigger is replayed on the recorded readings."""
    day = pd.Timestamp(date).normalize()
    S, E, in_data, weather, evs, cls, notes = _scenario(eng, day, weather, extra_events, extra_closures)
    lam, lam0, p, over, evload = _risk_tables(eng, S, E, weather, evs, cls)
    idx, keys = lam.index, eng.keys
    cap = eng.ceiling.values
    flag = (p >= 0.15) | (evload >= np.maximum(30, 0.3 * lam0.values))
    windows = []
    for j, k in enumerate(keys):
        on = np.where(flag[:, j])[0]
        if len(on) == 0:
            continue
        groups, cur = [], [on[0]]
        for i in on[1:]:
            if i - cur[-1] <= 2:
                cur.append(i)
            else:
                groups.append(cur); cur = [i]
        groups.append(cur)
        for g in groups:
            a, b = g[0], g[-1]
            sc = over[a:b + 1, j].sum() + 0.2 * evload[a:b + 1, j].sum()
            windows.append((sc, j, a, b))
    windows.sort(reverse=True)
    chosen, per_station = [], {}
    for sc, j, a, b in windows:
        if per_station.get(j, 0) >= 2:
            continue
        chosen.append((sc, j, a, b)); per_station[j] = per_station.get(j, 0) + 1
        if len(chosen) >= top:
            break
    obs = eng.flows.reindex(idx) if in_data else None
    cov = band_coverage(eng)
    watch = []
    for sc, j, a, b in sorted(chosen, key=lambda x: (idx[x[2]], -x[0])):
        k = keys[j]
        seg = slice(max(a - 1, 0), min(b + 2, len(idx)))            # one slot of margin on both sides
        t_idx = idx[seg]
        L, L0 = lam.values[seg, j], lam0.values[seg, j]
        hi90, nhi90 = quantile(L, cap[j], 0.95), quantile(L0, cap[j], 0.95)
        ipk = int(np.argmax(L))
        tpk = t_idx[ipk]
        reason = _event_reason(eng, evs, k, idx[a + int(np.argmax(evload[a:b + 1, j]))]) or \
            ('peak hour' + (' in rain' if _rain(eng, weather, idx[a], idx[b]) > 0.2 else ''))
        near = 0.9 * cap[j]
        lines = ', '.join(_lines_of(eng, k))
        triggers = [
            {'id': 'prepare', 'when': f"before {t_idx[0] - MIN15:%H:%M}",
             'if': 'always (planned)', 'then': f'position staff at {k} and prepare announcements',
             'indicator': f"forecast: up to {p[a:b + 1, j].max():.0%} chance per slot of reaching the ceiling ({int(cap[j])}); "
                          f"expected {int(eng.expected_reading(L[ipk], cap[j]))} at {tpk:%H:%M}"},
            {'id': 'surge', 'if': f"two consecutive readings above the normal level's upper 90 % bound "
                                  f"({int(nhi90.min())}-{int(nhi90.max())} depending on the slot)",
             'then': 'crowd bigger than a normal day: send staff to platforms and stairs, start announcements',
             'indicator': 'reading vs normal upper bound (a normal day exceeds it 1 time in 20, twice in a row 1 in 400)',
             'thresholds': {t.strftime('%H:%M'): int(v) for t, v in zip(t_idx, nhi90)}},
            {'id': 'above_plan', 'if': f"two consecutive readings above the forecast's upper 90 % bound "
                                       f"({int(hi90.min())}-{int(hi90.max())})",
             'then': f'demand above the plan: alert the control room, ask for extra trains on {lines}, re-check events and closures',
             'indicator': 'reading vs forecast upper bound (the plan already includes known events and weather)',
             'thresholds': {t.strftime('%H:%M'): int(v) for t, v in zip(t_idx, hi90)}},
            {'id': 'near_capacity', 'if': f'a reading of {int(near)} or more (90 % of the ceiling {int(cap[j])})',
             'then': 'meter access at the entrances, hold passengers at street level, keep exits clear',
             'indicator': 'reading vs ceiling'},
        ]
        rel = [c for c in cls if c['start'] and c['start'] <= t_idx[-1] + pd.Timedelta('1h') and c['end'] >= t_idx[0] - pd.Timedelta('1h')
               and (k in c['stations'] or (c['line'] and c['line'] in _lines_of(eng, k)))]
        alt = [(s2, d) for s2, d in eng.stations_near(eng.stations.at[k, 'latitude'], eng.stations.at[k, 'longitude'], 1.2)
               if s2 != k and set(_lines_of(eng, s2)) - set(_lines_of(eng, k))][:2]
        alt_txt = ', '.join(f'{s2} ({", ".join(_lines_of(eng, s2))}, {d} km)' for s2, d in alt) or 'replacement buses'
        triggers.append({'id': 'closure', 'if': ('closure in force: ' + '; '.join(c['description'] for c in rel)) if rel
                         else f'a closure or suspension is announced at {k} or on {lines}',
                         'then': f'inform passengers of the alternative: {alt_txt}; staff the transfer points',
                         'indicator': 'closure status'})
        item = {'station': k, 'lines': _lines_of(eng, k), 'window': f'{t_idx[0]:%H:%M}-{t_idx[-1] + MIN15:%H:%M}',
                'reason': reason, 'ceiling': int(cap[j]), 'triggers': triggers}
        if obs is not None:
            rec = obs[k].reindex(t_idx).values
            fired = {}
            for trg, thr in (('surge', nhi90), ('above_plan', hi90)):
                over2 = [i for i in range(1, len(rec)) if rec[i] > thr[i] and rec[i - 1] > thr[i - 1]]
                fired[trg] = t_idx[over2[0]].strftime('%H:%M') if over2 else None
            nc = [i for i in range(len(rec)) if rec[i] >= near]
            fired['near_capacity'] = t_idx[nc[0]].strftime('%H:%M') if nc else None
            item['replay_on_recorded_data'] = {'readings': {t.strftime('%H:%M'): int(v) for t, v in zip(t_idx, rec)},
                                               'would_have_fired_at': fired}
        watch.append(item)
    tot = eng.expected_reading(lam.values, cap[None, :]).sum()
    tot0 = eng.expected_reading(lam0.values, cap[None, :]).sum()
    out = {'date': day, 'weekday': day.day_name(), 'in_data': in_data, 'notes': notes,
           'assumptions': {'weather': weather or 'recorded', 'extra_events': extra_events or [], 'extra_closures': extra_closures or []},
           'network_expected_passengers': round(tot), 'network_normal_passengers': round(tot0),
           'watch': watch,
           'threshold_basis': f"ranges from the model's noise law; on the training summer the 80 % range held {cov['80']:.0%} "
                              f"of real readings and the 90 % range {cov['90']:.0%}; a reading above the 90 % upper bound "
                              f"happens {cov['share_above_90_upper']:.0%} of the time on a normal day"}
    if in_data:
        out['network_recorded_passengers'] = int(np.nansum(eng.flows.reindex(idx).values))
    return A._py(out)


# ------------------------------------------------------------------------------------------ staff plan
def staff_plan(eng: UBahnEngine, date: str, staff: int = 10, block_hours: float = 2.0, weather: dict | None = None,
               extra_events: list | None = None, extra_closures: list | None = None):
    """Where to put `staff` additional people, beyond the usual peak-hour staffing: each person covers a station
    for `block_hours`. The value of a block is the demand above capacity that a normal dry day (18 degC, no event)
    would not have, plus a share of event crowds; a second person on the same station and time counts half."""
    day = pd.Timestamp(date).normalize()
    S, E, in_data, weather, evs, cls, notes = _scenario(eng, day, weather, extra_events, extra_closures)
    lam, lam0, p, over, evload = _risk_tables(eng, S, E, weather, evs, cls)
    ref = eng.expected(S, E, weather={'tmean': 18.0, 'prcp': 0.0}, events=[], closures=cls).values
    cap = eng.ceiling.values[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        over_ref = np.where(ref > 0, ref * np.exp(-cap / np.maximum(ref, 1e-9)), 0)
    idx, keys = lam.index, eng.keys
    val = np.maximum(over - over_ref, 0) + 0.3 * evload
    B = max(int(block_hours * 4), 1)
    cover = np.zeros_like(val)
    picks = []
    for _ in range(int(staff)):
        weighted = val * (0.5 ** cover)
        cw = np.vstack([np.zeros((1, val.shape[1])), np.cumsum(weighted, axis=0)])
        block = cw[B:] - cw[:-B]
        i, j = np.unravel_index(np.argmax(block), block.shape)
        if block[i, j] <= 0:
            break
        cover[i:i + B, j] += 1
        picks.append((j, i))
    # merge overlapping picks of the same station into one shift
    shifts = []
    for j in sorted(set(pj for pj, _ in picks)):
        ivs = sorted((i, i + B) for pj, i in picks if pj == j)
        cur = [ivs[0][0], ivs[0][1], 1]
        for a, b in ivs[1:]:
            if a < cur[1]:
                cur[1] = max(cur[1], b); cur[2] += 1
            else:
                shifts.append((j, *cur)); cur = [a, b, 1]
        shifts.append((j, *cur))
    rows = []
    for j, a, b, n in shifts:
        k = keys[j]
        b = min(b, len(idx))
        rows.append({'station': k, 'lines': _lines_of(eng, k), 'from': idx[a].strftime('%H:%M'),
                     'to': (idx[b - 1] + MIN15).strftime('%H:%M'), 'staff': n,
                     'extra_passengers_above_capacity_vs_normal_day': round(float(np.maximum(over - over_ref, 0)[a:b, j].sum())),
                     'event_crowd': round(float(evload[a:b, j].sum())),
                     'peak_chance_at_ceiling': round(float(p[a:b, j].max()), 2),
                     'reason': _event_reason(eng, evs, k, idx[a + int(np.argmax(evload[a:b, j]))]) or
                               ('rain and peak demand' if _rain(eng, weather, idx[a], idx[b - 1]) > 0.2 else 'peak demand above a normal day')})
    rows.sort(key=lambda r: (r['from'], r['station']))
    return A._py({'date': day, 'weekday': day.day_name(), 'staff_available': staff, 'staff_assigned': int(sum(r['staff'] for r in rows)),
                  'block_hours': block_hours, 'notes': notes, 'roster': rows,
                  'method': 'additional staff beyond the usual peak-hour staffing: each person goes where the demand above capacity '
                            'exceeds a normal dry day the most (event crowds count too); a second person on the same station and time '
                            'counts half; overlapping assignments at a station are merged into one shift'})


# ------------------------------------------------------------------------------------------ daily brief
def daily_brief(eng: UBahnEngine, date: str, weather: dict | None = None, extra_events: list | None = None,
                extra_closures: list | None = None, staff: int = 10, assumptions: dict | None = None):
    """Proactive brief for a service day, without a question: situation, risks, staff and triggers."""
    day = pd.Timestamp(date).normalize()
    S, E, in_data, w_used, evs, cls, notes = _scenario(eng, day, weather, extra_events, extra_closures)
    plan = action_plan(eng, date, weather, extra_events, extra_closures, top=6)
    staffp = staff_plan(eng, date, staff, 2.0, weather, extra_events, extra_closures)
    reinf = reinforcement_plan(eng, date, weather, extra_events, extra_closures, top=6, assumptions=assumptions)
    if w_used is None:
        wx = eng.weather.loc[S:E]
        wtxt = (f"{wx.temp.min():.0f}-{wx.temp.max():.0f} degC, rain {wx.prcp.sum() / 4:.1f} mm"
                + (f" (up to {wx.prcp.max():.1f} mm/h)" if wx.prcp.max() > 0 else ', dry'))
    else:
        wtxt = f"scenario: {w_used.get('tmean')} degC daily mean, rain {w_used.get('prcp', 0)} mm/h"
    day_events = [e for e in evs if S - pd.Timedelta('2h') <= _to_dt(e['start']) <= E]
    day_closures = [c for c in cls if c['start'] and S - pd.Timedelta('1h') <= c['start'] <= E]
    heads = [f"{day:%A %d %B %Y}: {wtxt}."]
    heads.append(f"{len(day_events)} event(s)" + (': ' + '; '.join(f"{e.get('name', 'event')} at {A._py(eng.geocode(e.get('venue') or e.get('address'))[0])} "
                                                                  f"({int(e['attendance'])}, {_to_dt(e['start']):%H:%M}-{_to_dt(e['end']):%H:%M})"
                                                                  for e in day_events[:5]) if day_events else '') + '.')
    heads.append(f"{len(day_closures)} closure(s)" + (': ' + '; '.join(f"{c['description']} ({c['start']:%H:%M}-{c['end']:%H:%M})"
                                                                     for c in day_closures[:5]) if day_closures else '') + '.')
    if plan['watch']:
        top = sorted(plan['watch'], key=lambda w: -float(w['triggers'][0]['indicator'].split('up to ')[1].split('%')[0]))[:3]
        heads.append('Highest risks: ' + '; '.join(f"{w['station']} {w['window']} ({w['reason']})" for w in top) + '.')
    heads.append(f"Network: about {plan['network_expected_passengers']:,} passengers expected (normal {plan['network_normal_passengers']:,})"
                 + (f", {plan['network_recorded_passengers']:,} recorded" if in_data else '') + '.')
    if reinf['windows']:
        heads.append('Reinforcement: ' + ' '.join(reinf['summary'][:3]))
    if staffp['roster']:
        heads.append(f"Staff plan ({staff} people): " + '; '.join(f"{r['staff']} at {r['station']} {r['from']}-{r['to']}" for r in staffp['roster'][:6]) + '.')
    key_triggers = []
    for w in plan['watch'][:4]:
        t = {x['id']: x for x in w['triggers']}
        key_triggers.append({'station': w['station'], 'window': w['window'], 'if': t['surge']['if'], 'then': t['surge']['then']})
        key_triggers.append({'station': w['station'], 'window': w['window'], 'if': t['near_capacity']['if'], 'then': t['near_capacity']['then']})
    return A._py({'date': day, 'headlines': heads, 'notes': notes, 'weather': wtxt,
                  'events': [{'name': e.get('name', 'event'), 'venue': eng.geocode(e.get('venue') or e.get('address'))[0],
                              'start': _to_dt(e['start']), 'end': _to_dt(e['end']), 'attendance': e['attendance']} for e in day_events],
                  'closures': [{'description': c['description'], 'start': c['start'], 'end': c['end']} for c in day_closures],
                  'watch': [{k: w[k] for k in ('station', 'window', 'reason', 'ceiling')} for w in plan['watch']],
                  'staff_plan': staffp['roster'], 'key_triggers': key_triggers,
                  'reinforcement': {'summary': reinf['summary'], 'windows': reinf['windows'], 'caution': reinf['caution']},
                  'replay': [{'station': w['station'], 'would_have_fired_at': w['replay_on_recorded_data']['would_have_fired_at']}
                             for w in plan['watch'] if 'replay_on_recorded_data' in w]})


# ------------------------------------------------------------------------------------------ snapshot
def snapshot(eng: UBahnEngine, at: str, top: int = 10):
    """The network at one 15-minute slot: biggest deviations from normal, stations at their ceiling, events in
    their arrival or departure phase, weather and closures in force."""
    t = _to_dt(at).floor('15min')
    if t not in eng.flows.index:
        return {'error': f'{t:%Y-%m-%d %H:%M} is not a recorded slot (data {eng.flows.index.min():%Y-%m-%d} -> {eng.flows.index.max():%Y-%m-%d}, '
                         f'no service 01:00-04:45)'}
    keys = eng.keys
    obs = eng.flows.loc[t, keys].values
    lam = eng.expected(t, t)
    lam0 = eng.expected(t, t, events=[])
    cap = eng.ceiling.values
    er = eng.expected_reading(lam.values[0], cap)
    er0 = eng.expected_reading(lam0.values[0], cap)
    dev = pd.DataFrame({'station': keys, 'recorded': obs, 'normal': er0.round(0), 'model': er.round(0), 'ceiling': cap})
    dev['above_normal'] = dev.recorded - dev.normal
    dev['ratio'] = (dev.recorded / dev.normal.clip(lower=1)).round(1)
    topdev = dev.sort_values('above_normal', ascending=False).head(top)
    atcap = dev[dev.recorded >= dev.ceiling].station.tolist()
    phases = []
    for e in eng._data_events(t - pd.Timedelta('3h'), t + pd.Timedelta('2h')):
        s0, e0 = _to_dt(e['start']), _to_dt(e['end'])
        ph = ('arrivals' if s0 - pd.Timedelta('90min') <= t <= s0 + MIN15 else 'in progress' if s0 < t < e0
              else 'departures' if e0 <= t <= e0 + pd.Timedelta('2h') else None)
        if ph:
            phases.append({'event': e['name'], 'venue': eng.geocode(e['venue'])[0], 'attendance': int(e['attendance']),
                           'phase': ph, 'stations': list(eng.venue_shares(e['venue'])[1])[:4]})
    wx = eng.weather.loc[t] if t in eng.weather.index else None
    cl = [c['description'] for c in eng._clist if c['start'] <= t < c['end']]
    reopened = [c['description'] for c in eng._clist if c['kind'] == 'station' and c['end'] <= t < c['end'] + pd.Timedelta('45min')]
    net_obs, net_norm = float(obs.sum()), float(er0.sum())
    facts = [f"{t:%A %d %b %Y, %H:%M}: the network recorded {int(net_obs):,} passengers in 15 minutes against about {int(net_norm):,} "
             f"expected for this day and weather without events ({net_obs / max(net_norm, 1) - 1:+.0%})."]
    for _, r in topdev.head(3).iterrows():
        facts.append(f"{r.station}: {int(r.recorded)} against about {int(r.normal)} expected without events (ceiling {int(r.ceiling)}).")
    if atcap:
        facts.append(f"{len(atcap)} station(s) at their ceiling: " + ', '.join(atcap[:6]) + ('...' if len(atcap) > 6 else '') + '.')
    if phases:
        facts.append('Events: ' + '; '.join(f"{p['event']} ({p['phase']})" for p in phases) + '.')
    if cl:
        facts.append('Closures in force: ' + '; '.join(c.rstrip('.') for c in cl) + '.')
    return A._py({'slot': t, 'facts': facts, 'largest_deviations': topdev.to_dict(orient='records'),
                  'stations_at_ceiling': atcap, 'events': phases,
                  'weather': None if wx is None else {'temp': float(wx.temp), 'rain_mm_h': float(wx.prcp), 'wind_kmh': float(wx.wspd)},
                  'closures_in_force': cl, 'reopened_in_last_45_min': reopened,
                  'network': {'recorded': net_obs, 'normal': net_norm}})


# ------------------------------------------------------------------------------------------ learning report
def engine_state(eng: UBahnEngine) -> dict:
    """What the fitted engine knows, in a form that can be compared after new data is injected."""
    ev = eng.events
    venues = sorted(set(ev.venue_key.dropna()))
    return {'flows_end': str(eng.flows.index.max()), 'flows_start': str(eng.flows.index.min()),
            'w': {k: float(v) for k, v in eng.w.items()}, 'ceiling': {k: float(v) for k, v in eng.ceiling.items()},
            'weather_params': {k: float(v) for k, v in eng.weather_params.items()},
            'venue_learned': {k: dict(v) for k, v in getattr(eng, 'venue_learned', {}).items()},
            'venue_rule': {v: eng.venue_shares(v)[1] for v in venues},
            'n_events': int(len(ev)), 'n_closures': int(len(eng._clist)), 'fitted_at': getattr(eng, 'fitted_at', None)}


def learning_report(before: dict, eng: UBahnEngine, save_to: str | None = None) -> dict:
    """Compare the engine before and after an injection: new period, venues whose crowd split was learned or
    changed, how the new days fit the model, stations whose level or ceiling moved, what could not be read."""
    after = engine_state(eng)
    old_end = pd.Timestamp(before['flows_end'])
    new_end = eng.flows.index.max()
    new_days = sorted(set(eng.service_day(eng.flows.index[eng.flows.index > old_end])))
    rep = {'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'), 'period_before': [before['flows_start'], before['flows_end']],
           'period_after': [str(eng.flows.index.min()), str(new_end)], 'new_service_days': [d.strftime('%Y-%m-%d') for d in new_days]}
    venues = []
    new_ev = eng.events[eng.events.start > old_end]
    venues_with_new_events = set(new_ev.venue_key.dropna())
    for v, sh in after['venue_learned'].items():
        old = before['venue_learned'].get(v)
        diag = getattr(eng, 'venue_learned_diag', {}).get(v, {})
        if old is None:
            rule = before['venue_rule'].get(v) or eng.rule_shares(v)
            venues.append({'venue': v, 'status': 'crowd split learned from the data', 'method': diag.get('method'),
                           'before (distance rule)': {k: round(x, 2) for k, x in sorted(rule.items(), key=lambda x: -x[1])},
                           'after (learned)': {k: round(x, 2) for k, x in sorted(sh.items(), key=lambda x: -x[1])}})
        elif v in venues_with_new_events:
            diff = max(abs(sh.get(k, 0) - old.get(k, 0)) for k in set(sh) | set(old))
            if diff > 0.05:
                venues.append({'venue': v, 'status': 'crowd split updated with the new events', 'method': diag.get('method'),
                               'before': {k: round(x, 2) for k, x in sorted(old.items(), key=lambda x: -x[1])},
                               'after': {k: round(x, 2) for k, x in sorted(sh.items(), key=lambda x: -x[1])}})
    for v in sorted(venues_with_new_events - set(after['venue_learned'])):
        venues.append({'venue': v, 'status': 'not enough evidence yet to learn its split: the distance rule is kept',
                       'rule': {k: round(x, 2) for k, x in sorted(eng.rule_shares(v).items(), key=lambda x: -x[1])}})
    rep['venue_splits'] = venues
    if new_days:
        cal = eng.calibration(new_days[0] + pd.Timedelta('5h'), new_end)
        rep['new_days_vs_model'] = cal
    ch_w = []
    for k, v in after['w'].items():
        o = before['w'].get(k)
        if o and abs(v / o - 1) > 0.05:
            ch_w.append({'station': k, 'level_change_pct': round((v / o - 1) * 100, 1)})
    rep['station_levels_changed_over_5pct'] = sorted(ch_w, key=lambda r: -abs(r['level_change_pct']))[:10]
    rep['ceilings_raised'] = [{'station': k, 'before': before['ceiling'].get(k), 'after': v}
                              for k, v in after['ceiling'].items() if before['ceiling'].get(k) and v > before['ceiling'][k]][:10]
    rep['weather_rule'] = {'before': before['weather_params'], 'after': after['weather_params']}
    st = eng.data_status()
    rep['events_added'] = after['n_events'] - before['n_events']
    rep['closures_added'] = after['n_closures'] - before['n_closures']
    rep['events_not_located'] = st['events_not_located']
    rep['closures_not_understood'] = st['closures_not_understood']
    summ = [f"New data: {len(new_days)} service day(s), {rep['events_added']} event listing(s), {rep['closures_added']} closure(s)."]
    for v in venues:
        if 'rule' in v:
            continue
        after_txt = ', '.join(f'{k} {x:.0%}' for k, x in list(v.get('after (learned)', v.get('after', {})).items())[:4])
        before_txt = ', '.join(f'{k} {x:.0%}' for k, x in list(v.get('before (distance rule)', v.get('before', {})).items())[:4])
        summ.append(f"{v['venue']}: {v['status']}: {after_txt} (before: {before_txt or 'unknown'}).")
    if new_days and 'new_days_vs_model' in rep:
        by = rep['new_days_vs_model']['by_day']
        summ.append('New days against the model (1.0 = as expected): ' + ', '.join(f'{d[5:]} {r:.2f}' for d, r in by.items()) + '.')
    if ch_w:
        summ.append(f"{len(ch_w)} station level(s) moved more than 5 %, e.g. " + ', '.join(f"{r['station']} {r['level_change_pct']:+.0f} %" for r in rep['station_levels_changed_over_5pct'][:3]) + '.')
    if rep['events_not_located']:
        summ.append(f"{len(rep['events_not_located'])} venue(s) could not be located: add them to venues_extra.csv.")
    if rep['closures_not_understood']:
        summ.append(f"{len(rep['closures_not_understood'])} closure description(s) not understood.")
    rep['summary'] = summ
    rep = A._py(rep)
    if save_to:
        with open(os.path.join(save_to, 'learning_report.json'), 'w', encoding='utf-8') as f:
            json.dump(rep, f, ensure_ascii=False, indent=1)
    return rep


def last_learning_report(eng: UBahnEngine) -> dict:
    path = os.path.join(eng.data_dir, 'learning_report.json')
    if not os.path.exists(path):
        return {'note': 'no injection has been made since the system was set up; the report appears after ubahn_ingest.py adds files'}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ================================================================================== v0.6 additions
# ------------------------------------------------------------------------------------------ reliability
def backtest_summary(eng: UBahnEngine) -> dict:
    """How well the model reproduces the recorded data (computed once per engine, ~2 s)."""
    if getattr(eng, '_backtest', None):
        return eng._backtest
    t0, t1 = eng.flows.index.min(), eng.flows.index.max()
    lam = eng.expected(t0, t1)
    base = eng.expected(t0, t1, events=[])
    F = eng.flows.reindex(lam.index).values
    cap = eng.ceiling.values[None, :]
    er = eng.expected_reading(lam.values, cap)
    h = lam.index.hour
    hourly = pd.Series(F.sum(axis=1)).groupby(h).sum() / pd.Series(er.sum(axis=1)).groupby(h).sum()
    evl = lam.values - base.values
    m = evl > 50
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(lam.values > 0, np.exp(-cap / np.maximum(lam.values, 1e-9)), 0)
    sd = eng.service_day(lam.index)
    d_obs = pd.Series(F.sum(axis=1), index=lam.index).groupby(sd).sum()
    d_exp = pd.Series(er.sum(axis=1), index=lam.index).groupby(sd).sum()
    out = {'period': f'{t0:%Y-%m-%d} -> {t1:%Y-%m-%d}', 'slots': int(F.shape[0]), 'stations': int(F.shape[1]),
           'total_observed_over_expected': round(float(F.sum() / er.sum()), 3),
           'hourly_observed_over_expected_range': [round(float(hourly.drop(0, errors='ignore').min()), 3),
                                                   round(float(hourly.drop(0, errors='ignore').max()), 3)],
           'event_slots_observed_over_expected': round(float(F[m].sum() / er[m].sum()), 3) if m.any() else None,
           'readings_at_ceiling': {'expected': int(p.sum()), 'observed': int((F >= cap).sum()),
                                   'correlation_across_stations': round(float(np.corrcoef(p.sum(axis=0), (F >= cap).sum(axis=0))[0, 1]), 3)},
           'daily_network_totals': {'correlation': round(float(np.corrcoef(d_obs, d_exp)[0, 1]), 3),
                                    'mean_absolute_error_pct': round(float((abs(d_obs - d_exp) / d_obs).mean() * 100), 2)}}
    eng._backtest = out
    return out


def reliability_report(eng: UBahnEngine) -> dict:
    """Evidence that the numbers can be trusted: calibration of the plausible ranges and backtest of the model."""
    cov = band_coverage(eng)
    return A._py({'forecast_ranges_calibration': {'80 % range holds': cov['80'], '90 % range holds': cov['90'],
                                                 'readings above the 90 % upper bound on normal days': cov['share_above_90_upper']},
                  'backtest': backtest_summary(eng), 'fitted_at': getattr(eng, 'fitted_at', None),
                  'how_answers_are_checked': 'every figure of an answer is matched against the tool results that produced it '
                                             '(verified, calculated from two verified figures, or flagged as not found)',
                  'limits': ['single 15-minute readings are very noisy (a reading can be a tenth or three times its expected level)',
                             'line suspensions leave no trace in the flows; rerouting advice is a planning assumption',
                             'the model reproduces the hackathon simulator, not real ridership']})


# ------------------------------------------------------------------------------------------ energy
def energy_stats(eng: UBahnEngine, lines: list | None = None, date_from: str | None = None, date_to: str | None = None,
                 days: str = 'all', by: str = 'total'):
    """Recorded energy per line over a period, with ridership (interchange stations counted for each line they
    serve), kWh per passenger, highest and lowest days, and how far each day was from what the ridership explains.
    by = total | day."""
    daily, _, _ = A._daily(eng)
    E = eng.energy.astype(float).copy()
    E.index = pd.to_datetime(E.index).normalize()
    sel = [L.upper() for L in lines] if lines else list(E.columns)
    bad = [L for L in sel if L not in E.columns]
    if bad:
        raise ValueError(f'no energy data for {bad} (lines: {list(E.columns)})')
    idx = E.index
    m = np.ones(len(idx), bool)
    if date_from:
        m &= idx >= pd.Timestamp(date_from).normalize()
    if date_to:
        m &= idx <= pd.Timestamp(date_to).normalize()
    dt = eng.daytype(idx)
    if days == 'weekday':
        m &= dt == 0
    elif days == 'saturday':
        m &= dt == 1
    elif days == 'sunday':
        m &= dt == 2
    elif days == 'weekend':
        m &= dt > 0
    elif days != 'all':
        raise ValueError('days must be all, weekday, saturday, sunday or weekend')
    d = idx[m]
    if len(d) == 0:
        return {'error': 'no energy data for these dates', 'energy_range': f'{idx.min():%Y-%m-%d} -> {idx.max():%Y-%m-%d}'}
    rows, per_day = {}, {}
    for L in sel:
        ks = [k for k in eng.keys if L in eng.stations.at[k, 'lines']]
        R = daily[ks].sum(axis=1).reindex(d)
        e = E.loc[d, L]
        b0, b1, sdv = eng.energy_model[L]
        expct = b0 + b1 * R
        resid = e - expct
        ok = R.notna()
        rows[L] = {'days': int(ok.sum()), 'total_mwh': float(e[ok].sum()), 'mean_mwh_per_day': float(e[ok].mean()),
                   'passengers_total': float(R[ok].sum()), 'kwh_per_passenger': float(e[ok].sum() * 1000 / R[ok].sum()) if ok.any() else None,
                   'highest_day': [f'{e[ok].idxmax():%Y-%m-%d}', float(e[ok].max())], 'lowest_day': [f'{e[ok].idxmin():%Y-%m-%d}', float(e[ok].min())],
                   'mean_gap_vs_ridership_model_mwh': float(resid[ok].mean()),
                   'most_unusual_day': [f'{resid[ok].abs().idxmax():%Y-%m-%d}', float(resid[ok][resid[ok].abs().idxmax()])] if ok.any() else None,
                   'usual_daily_gap_sd_mwh': float(sdv)}
        if by == 'day':
            per_day[L] = [{'date': f'{t:%Y-%m-%d}', 'weekday': t.day_name()[:3], 'mwh': float(e[t]),
                           'passengers': float(R[t]) if ok[t] else None, 'expected_from_ridership_mwh': round(float(expct[t]), 1) if ok[t] else None}
                          for t in d[:45]]
    out = {'period': [f'{d.min():%Y-%m-%d}', f'{d.max():%Y-%m-%d}'], 'days_filter': days, 'by_line': rows,
           'network_total_mwh': float(sum(r['total_mwh'] for r in rows.values())),
           'note': 'energy is almost entirely explained by ridership (about 83 % fixed); a day far from the ridership model '
                   'by more than 2-3 times the usual gap is unusual'}
    if per_day:
        out['by_day'] = per_day
    return A._py(out)


# ------------------------------------------------------------------------------------------ passenger messages
REASONS = {  # english key: (english, german, french)
    'safety inspection': ('due to a safety inspection', 'wegen einer Sicherheitsprüfung', 'en raison d’une inspection de sécurité'),
    'switch replacement': ('due to switch replacement works', 'wegen einer Weichenerneuerung', 'en raison du remplacement d’un aiguillage'),
    'track maintenance': ('due to track works', 'wegen Gleisarbeiten', 'en raison de travaux sur la voie'),
    'track works': ('due to track works', 'wegen Gleisarbeiten', 'en raison de travaux sur la voie'),
    'power system maintenance': ('due to power supply works', 'wegen Arbeiten an der Stromversorgung', 'en raison de travaux sur l’alimentation électrique'),
    'signal failure': ('due to a signal failure', 'wegen einer Signalstörung', 'en raison d’une panne de signalisation'),
    'escalator works': ('due to escalator works', 'wegen Arbeiten an der Rolltreppe', 'en raison de travaux sur l’escalier mécanique'),
    'police operation': ('due to a police operation', 'wegen eines Polizeieinsatzes', 'en raison d’une intervention de police'),
    'technical fault': ('due to a technical fault', 'wegen einer technischen Störung', 'en raison d’un incident technique'),
}
DEFAULT_REASON = ('due to a disruption', 'wegen einer Betriebsstörung', 'en raison d’une perturbation')


def _reason(desc):
    d = desc.lower()
    for k, v in REASONS.items():
        if k in d:
            return v
    return DEFAULT_REASON


def _tidy(msgs):
    return {lang: {ch: re.sub(r'\.\.(?=\s|$)', '.', txt) for ch, txt in d.items()} for lang, d in msgs.items()}


def _nm(st):
    return st.replace(' Bhf', '').replace(' (U2)', '').replace(' (U6)', '')


def _join(items, word):
    items = [i for i in items if i]
    return items[0] if len(items) == 1 else (', '.join(items[:-1]) + f' {word} ' + items[-1]) if items else ''


def passenger_messages(eng: UBahnEngine, closure: str | None = None, start: str | None = None, end: str | None = None,
                       event: str | None = None, date: str | None = None, venue: str | None = None,
                       event_start: str | None = None, event_end: str | None = None, until_label: str | None = None):
    """Ready-to-use passenger messages in German, English and French (platform announcement, display text,
    app text), built from the network data, for a closure (its description, start and end, or a closure of the
    data found by `closure` text and `date`) or an event (a name found in the events data on `date`, or a
    venue with event_start and event_end)."""
    out = {}
    if closure:
        c = None
        if start and end:
            c = eng.parse_closure(closure, start, end)
        else:
            for cc in eng._clist:
                if closure.lower() in cc['description'].lower() and (not date or eng.service_day(pd.DatetimeIndex([cc['start']]))[0] == pd.Timestamp(date).normalize()):
                    c = cc
                    break
        if c is None or c['start'] is None:
            return {'error': 'closure not found: give its description with start and end'}
        until = until_label or c['end'].strftime('%H:%M')
        en_r, de_r, fr_r = _reason(c['description'])
        if c['kind'] == 'line' and c['stations']:
            ro = eng.reroute_options(c['line'], c['stations'][0], c['stations'][-1])
            a, b = _nm(ro['section'][0]), _nm(ro['section'][-1])
            alts = {}
            for s in ro['interior_without_service'] + [ro['section'][0], ro['section'][-1]]:
                ls = ro['per_station_alternatives'][s]['other_lines_here']
                if ls:
                    alts.setdefault(_nm(s), '/'.join(ls))
            alts = list(alts.items())[:3]
            sb = [_nm(s) for s in ro['interior_without_service'] if A._sbahn(eng, s)][:1]
            L = c['line']
            en_alt = _join([f'the {l} at {s}' for s, l in alts] + [f'the S-Bahn at {s}' for s in sb], 'or')
            de_alt = _join([f'die {l} ab {s}' for s, l in alts] + [f'die S-Bahn ab {s}' for s in sb], 'oder')
            fr_alt = _join([f'la {l} à {s}' for s, l in alts] + [f'le S-Bahn à {s}' for s in sb], 'ou')
            short = ', '.join([f'{l} ({s})' for s, l in alts] + [f'S-Bahn ({s})' for s in sb])
            parts = [p.split(' - ') for p in ro['line_split_into']]
            run_en = ' and '.join(f'between {_nm(x)} and {_nm(y)}' for x, y in parts)
            run_de = ' sowie '.join(f'zwischen {_nm(x)} und {_nm(y)}' for x, y in parts)
            run_fr = ', et '.join(f'entre {_nm(x)} et {_nm(y)}' for x, y in parts)
            out = {
                'en': {'announcement': f'Attention please: line {L} is suspended between {a} and {b} {en_r}, expected until {until}. '
                                       + (f'Please use {en_alt}. ' if en_alt else 'Please follow staff instructions. ') + 'We apologise for the inconvenience.',
                       'display': f'{L} {a} – {b}: no service until {until}' + (f' · {short}' if short else ''),
                       'app': f'{L} is suspended between {a} and {b} {en_r} until about {until}. {L} trains still run {run_en}. '
                              + (f'Alternatives: {en_alt}.' if en_alt else '')},
                'de': {'announcement': f'Achtung: Die Linie {L} ist zwischen {a} und {b} {de_r} unterbrochen, voraussichtlich bis {until} Uhr. '
                                       + (f'Bitte nutzen Sie {de_alt}. ' if de_alt else 'Bitte folgen Sie den Hinweisen des Personals. ') + 'Wir bitten um Entschuldigung.',
                       'display': f'{L} {a} – {b}: kein Zugverkehr bis {until} Uhr' + (f' · {short}' if short else ''),
                       'app': f'Die {L} ist zwischen {a} und {b} {de_r} bis ca. {until} Uhr unterbrochen. Die {L} fährt weiterhin {run_de}. '
                              + (f'Alternativen: {de_alt}.' if de_alt else '')},
                'fr': {'announcement': f'Votre attention s’il vous plaît : la ligne {L} est interrompue entre {a} et {b} {fr_r}, jusqu’à {until} environ. '
                                       + (f'Merci d’emprunter {fr_alt}. ' if fr_alt else 'Merci de suivre les consignes du personnel. ') + 'Nous vous prions de nous excuser.',
                       'display': f'{L} {a} – {b} : pas de trafic jusqu’à {until}' + (f' · {short}' if short else ''),
                       'app': f'La {L} est interrompue entre {a} et {b} {fr_r} jusqu’à {until} environ. La {L} circule toujours {run_fr}. '
                              + (f'Alternatives : {fr_alt}.' if fr_alt else '')}}
        elif c['kind'] in ('station', 'platform') and c['stations']:
            k = c['stations'][0]
            nb = [_nm(n) for n in eng.graph.neighbors(k)][:2]
            s = _nm(k)
            if c['kind'] == 'station':
                out = {
                    'en': {'announcement': f'Attention please: {s} station is closed {en_r} until {until}. Trains do not stop there. Please use {_join(nb, "or")}.',
                           'display': f'{s} closed until {until} · use {_join(nb, "or")}',
                           'app': f'{s} is closed {en_r} until about {until}; trains pass without stopping. Nearest stations: {_join(nb, "and")}.'},
                    'de': {'announcement': f'Achtung: Der Bahnhof {s} ist {de_r} bis {until} Uhr geschlossen. Die Züge halten dort nicht. Bitte nutzen Sie {_join(nb, "oder")}.',
                           'display': f'{s} geschlossen bis {until} Uhr · bitte {_join(nb, "oder")} nutzen',
                           'app': f'Der Bahnhof {s} ist {de_r} bis ca. {until} Uhr geschlossen; die Züge fahren ohne Halt durch. Nächste Bahnhöfe: {_join(nb, "und")}.'},
                    'fr': {'announcement': f'Votre attention s’il vous plaît : la station {s} est fermée {fr_r} jusqu’à {until}. Les trains ne s’y arrêtent pas. Merci d’utiliser {_join(nb, "ou")}.',
                           'display': f'{s} fermée jusqu’à {until} · utilisez {_join(nb, "ou")}',
                           'app': f'La station {s} est fermée {fr_r} jusqu’à {until} environ ; les trains la traversent sans s’arrêter. Stations les plus proches : {_join(nb, "et")}.'}}
            else:
                pm = re.search(r'platform\s+(\w+)', c['description'], re.I)
                pn = pm.group(1) if pm else None
                out = {
                    'en': {'announcement': f'Attention please: at {s}, {"platform " + pn if pn else "one platform"} is closed {en_r} until {until}. Please follow the signs and staff instructions.',
                           'display': f'{s}: {"platform " + pn if pn else "a platform"} closed until {until}',
                           'app': f'At {s}, {"platform " + pn if pn else "one platform"} is closed {en_r} until about {until}. Allow extra time and follow the signs.'},
                    'de': {'announcement': f'Achtung: Am Bahnhof {s} ist {"Bahnsteig " + pn if pn else "ein Bahnsteig"} {de_r} bis {until} Uhr gesperrt. Bitte folgen Sie der Beschilderung und den Hinweisen des Personals.',
                           'display': f'{s}: {"Bahnsteig " + pn if pn else "ein Bahnsteig"} gesperrt bis {until} Uhr',
                           'app': f'Am Bahnhof {s} ist {"Bahnsteig " + pn if pn else "ein Bahnsteig"} {de_r} bis ca. {until} Uhr gesperrt. Bitte planen Sie mehr Zeit ein.'},
                    'fr': {'announcement': f'Votre attention s’il vous plaît : à {s}, {"le quai " + pn if pn else "un quai"} est fermé {fr_r} jusqu’à {until}. Merci de suivre la signalisation et les consignes du personnel.',
                           'display': f'{s} : {"quai " + pn if pn else "un quai"} fermé jusqu’à {until}',
                           'app': f'À {s}, {"le quai " + pn if pn else "un quai"} est fermé {fr_r} jusqu’à {until} environ. Prévoyez plus de temps.'}}
        else:
            return {'error': 'closure understood only partly: ' + '; '.join(c['issues'])}
        out = _tidy(out)
        return A._py({'situation': c['description'], 'start': c['start'], 'end': c['end'], 'messages': out,
                      'note': 'templates filled from the network data; check the times before broadcasting'})
    if event or venue:
        s0 = e0 = None
        vkey, shares, name = None, {}, event or venue
        if event and date and not (event_start and event_end):
            ev = eng.events[eng.service_day(eng.events.start) == pd.Timestamp(date).normalize()]
            ev = ev[ev.event_name.str.lower().str.contains(event.lower(), regex=False) | ev.venue_name.fillna('').str.lower().str.contains(event.lower(), regex=False)
                    | ev.address.str.lower().str.contains(event.lower(), regex=False)]
            if ev.empty:
                return {'error': f'no event matching "{event}" on {date}'}
            r = ev.iloc[0]
            s0, e0, name = r.start, ev[ev.start == r.start].end.max(), r.event_name
            vkey, shares = eng.venue_shares(r.address + ' ' + str(r.venue_name if isinstance(r.venue_name, str) else ''))
        else:
            if not (venue and event_start and event_end):
                return {'error': 'give an event name and date, or a venue with event_start and event_end'}
            s0, e0 = _to_dt(event_start), _to_dt(event_end)
            vkey, shares = eng.venue_shares(venue)
        if not shares:
            return {'error': 'venue could not be located'}
        top = [k for k, _ in sorted(shares.items(), key=lambda x: -x[1])]
        main = top[:2]
        alt = [k for k in top[2:] if shares[k] >= 0.05][:1]
        if not alt:
            _, vlat, vlon = eng.geocode(venue or (r.address if event else ''))
            lines_main = set(sum([eng.stations.at[k, 'lines'] for k in main], []))
            alt = [k for k, dkm in eng.stations_near(eng.stations.at[main[0], 'latitude'], eng.stations.at[main[0], 'longitude'], 2.6)
                   if k not in shares and not (set(eng.stations.at[k, 'lines']) & lines_main)][:1]
        vn = (vkey or venue or '').split(' (')[0].split(' / ')[0]
        mst, ast = [_nm(k) for k in main], [_nm(k) for k in alt]
        a0, a1 = (s0 - pd.Timedelta('90min')).strftime('%H:%M'), s0.strftime('%H:%M')
        d0, d1 = e0.strftime('%H:%M'), (e0 + pd.Timedelta('2h')).strftime('%H:%M')
        out = {
            'en': {'before': f'Heavy crowds are expected at {_join(mst, "and")} from {a0} before the event at {vn} ({a1}). Please allow extra time.',
                   'after': f'After the event at {vn}, heavy crowds are expected at {_join(mst, "and")} between {d0} and {d1}. Please allow extra time'
                            + (f' and consider {_join(ast, "or")}' if ast else '') + ', and follow staff instructions.',
                   'display': f'{_join(mst, "/")}: heavy crowds {d0}–{d1}' + (f' · also use {_join(ast, "or")}' if ast else '')},
            'de': {'before': f'Vor der Veranstaltung ({vn}, Beginn {a1} Uhr) erwarten wir ab {a0} Uhr starken Andrang an den Bahnhöfen {_join(mst, "und")}. Bitte planen Sie mehr Zeit ein.',
                   'after': f'Nach der Veranstaltung ({vn}) erwarten wir zwischen {d0} und {d1} Uhr starken Andrang an den Bahnhöfen {_join(mst, "und")}. Bitte planen Sie mehr Zeit ein'
                            + (f' und nutzen Sie auch {_join(ast, "oder")}' if ast else '') + '. Bitte folgen Sie den Hinweisen des Personals.',
                   'display': f'{_join(mst, "/")}: starker Andrang {d0}–{d1} Uhr' + (f' · auch {_join(ast, "oder")} nutzen' if ast else '')},
            'fr': {'before': f'Avant l’événement ({vn}, début à {a1}), une forte affluence est attendue à partir de {a0} aux stations {_join(mst, "et")}. Prévoyez plus de temps.',
                   'after': f'Après l’événement ({vn}), une forte affluence est attendue aux stations {_join(mst, "et")} entre {d0} et {d1}. Prévoyez plus de temps'
                            + (f' et pensez à {_join(ast, "ou")}' if ast else '') + ', et suivez les consignes du personnel.',
                   'display': f'{_join(mst, "/")} : forte affluence {d0}–{d1}' + (f' · utilisez aussi {_join(ast, "ou")}' if ast else '')}}
        out = _tidy(out)
        return A._py({'situation': f'{name if event else "Event"} at {vn}', 'start': s0, 'end': e0, 'stations': main, 'alternative': alt,
                      'messages': out, 'note': 'templates filled from the network data; the end time is the dataset estimate'})
    return {'error': 'give a closure or an event'}


# ================================================================================== v0.7 addition
# ------------------------------------------------------------------------------------------ reinforcement across modes
MODE_ASSUMPTIONS = {
    # planning assumptions: none of these are in the dataset (it only records U-Bahn station flows)
    'ubahn_trains_per_hour': [[5, 6, 6], [6, 9, 15], [9, 15, 12], [15, 19, 15], [19, 21, 12], [21, 29, 6]],  # [from h, to h, trains/h]
    'ubahn_max_trains_per_hour': 20,          # about a 3-minute headway
    'sbahn_walk_km': 1.0,                     # S-Bahn station within walking distance of the crowded station
    'sbahn_extra_trains_max': 4,              # extra S-Bahn trains per hour that could be requested
    'sbahn_spare_per_train': 400,             # passengers an extra S-Bahn train can take at that station
    'bus_capacity': 90,                       # articulated bus
    'bus_speed_kmh': 15, 'bus_loading_min': 10, 'bus_max_km': 2.6, 'buses_max': 20,
    'taxi_pax_per_hour': 4, 'taxis_max': 40,  # 2 trips an hour, 2 passengers each
    'bike_pax_per_hour': 1, 'bikes_max': 80, 'bike_max_rain_mm_h': 0.2,
    'target_reduction': 0.9,                  # aim: remove 90 % of the demand above capacity beyond a normal day
    'min_extra_per_slot': 5,                  # a slot counts when it has 5+ extra passengers above capacity
    'min_extra_per_window': 60,               # and a window only when it adds up to 60+ extra passengers
    'walk_max_km': 0.8,                       # below this distance, guide passengers on foot instead of running buses
}


def _trains_per_hour(hour, table):
    h = hour if hour >= 5 else hour + 24
    for a, b, f in table:
        if a <= h < b:
            return f
    return 6


def _sbahn_options(eng, st, walk_km):
    """S-Bahn access for a station: the station itself (S+U), S+U stations within walking distance, and
    S-Bahn-only stations listed in <data>/sbahn_extra.csv (name, lat, lon, lines)."""
    lat, lon = eng.stations.at[st, 'latitude'], eng.stations.at[st, 'longitude']
    out = []
    if A._sbahn(eng, st):
        out.append((st, 0.0, 'S+U interchange'))
    for k, d in eng.stations_near(lat, lon, walk_km):
        if k != st and A._sbahn(eng, k):
            out.append((k, d, 'S+U interchange'))
    extra = os.path.join(eng.data_dir, 'sbahn_extra.csv')
    if os.path.exists(extra):
        try:
            sx = pd.read_csv(extra)
            from src.engine import _hav
            for _, r in sx.iterrows():
                d = float(_hav(lat, lon, float(r['lat']), float(r['lon'])))
                if d <= walk_km:
                    out.append((str(r['name']), round(d, 2), f"S-Bahn {r.get('lines', '')}".strip()))
        except Exception:
            pass
    return sorted(out, key=lambda x: x[1])


def reinforcement_plan(eng: UBahnEngine, date: str, weather: dict | None = None, extra_events: list | None = None,
                       extra_closures: list | None = None, top: int = 6, assumptions: dict | None = None):
    """When demand exceeds what a normal day asks of the U-Bahn, how to absorb it across modes: more U-Bahn trains
    (platform capacity grows with frequency, up to a maximum), extra S-Bahn trains where an S-Bahn station is within
    walking distance, shuttle buses to a station on another line, bikes (dry daytime) and taxis (night first), and
    what is left to meter at the entrances. Quantities rest on the planning assumptions returned with the plan."""
    a = {**MODE_ASSUMPTIONS, **(assumptions or {})}
    day = pd.Timestamp(date).normalize()
    S, E, in_data, weather, evs, cls, notes = _scenario(eng, day, weather, extra_events, extra_closures)
    idx = eng.slots(S, E)
    lam = eng.expected(S, E, weather=weather, events=evs, closures=cls).values
    # a normal day for comparison: same temperature, no rain, no event
    if weather is not None:
        t_ref = float(weather.get('tmean', 18.0))
    else:
        tt = eng._daily_temp(idx)
        t_ref = float(np.nanmean(tt)) if np.isfinite(tt).any() else 18.0
    ref = eng.expected(S, E, weather={'tmean': t_ref, 'prcp': 0.0}, events=[], closures=cls).values
    cap = eng.ceiling.values[None, :]
    er_all = eng.expected_reading(lam, cap)

    def above(l, c):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(l > 0, l * np.exp(-c / np.maximum(l, 1e-9)), 0.0)
    over, over_ref = above(lam, cap), above(ref, cap)
    extra = np.maximum(over - over_ref, 0)
    flag = extra >= a['min_extra_per_slot']
    wins = []
    for j in range(len(eng.keys)):
        on = np.where(flag[:, j])[0]
        if len(on) == 0:
            continue
        cur = [on[0]]
        for i in list(on[1:]) + [None]:
            if i is not None and i - cur[-1] <= 2:
                cur.append(i)
                continue
            tot = float(extra[cur[0]:cur[-1] + 1, j].sum())
            if tot >= a['min_extra_per_window']:
                wins.append((tot, j, cur[0], cur[-1]))
            if i is not None:
                cur = [i]
    wins.sort(reverse=True)
    wins = wins[:top]
    plans, line_needs = [], {}
    for score, j, i0, i1 in sorted(wins, key=lambda w: (w[2], -w[0])):
        k = eng.keys[j]
        c = float(cap[0, j])
        L, Ov, Oref = lam[i0:i1 + 1, j], over[i0:i1 + 1, j], over_ref[i0:i1 + 1, j]
        hours = (i1 - i0 + 1) / 4
        f0 = _trains_per_hour(idx[i0].hour, a['ubahn_trains_per_hour'])
        fmax = max(a['ubahn_max_trains_per_hour'], f0)
        target = Oref.sum() + (1 - a['target_reduction']) * (Ov.sum() - Oref.sum())
        df = 0
        for df in range(0, fmax - f0 + 1):
            if above(L, c * (1 + df / f0)).sum() <= target:
                break
        left = above(L, c * (1 + df / f0)).sum()
        resid_h = max(left - Oref.sum(), 0) / hours
        lines = _lines_of(eng, k)
        measures = []
        if df > 0:
            measures.append({'mode': 'U-Bahn', 'action': f"+{df} trains per hour on {'/'.join(lines)} ({f0} -> {f0 + df} per hour)",
                             'quantity': df, 'absorbs_per_hour': round((Ov.sum() - left) / hours),
                             'basis': 'platform capacity grows in proportion to train frequency'})
            for ln in lines:
                line_needs.setdefault(ln, []).append((idx[i0], idx[i1] + MIN15, df, f0))
        rain = _rain(eng, weather, idx[i0], idx[i1])
        night = idx[i0].hour >= 23 or idx[i0].hour < 5
        sb = _sbahn_options(eng, k, a['sbahn_walk_km'])
        if sb and resid_h > 0 and a['sbahn_extra_trains_max'] > 0:
            take = min(resid_h, a['sbahn_extra_trains_max'] * a['sbahn_spare_per_train'])
            n = int(np.ceil(take / a['sbahn_spare_per_train']))
            if sb[0][1] == 0:
                where = f'the S-Bahn platforms at {k}'
            else:
                where = ', '.join(f'{s_} ({d_} km)' for s_, d_, _ in sb[:2])
            measures.append({'mode': 'S-Bahn', 'action': f'+{n} S-Bahn train(s) per hour and guide passengers to {where}',
                             'quantity': n, 'absorbs_per_hour': round(take), 'basis': f"{a['sbahn_spare_per_train']} passengers per extra train"})
            resid_h -= take
        if resid_h > 0:
            alts = [(s2, d) for s2, d in eng.stations_near(eng.stations.at[k, 'latitude'], eng.stations.at[k, 'longitude'], a['bus_max_km'])
                    if s2 != k and not (set(_lines_of(eng, s2)) & set(lines))]
            for s2, d in alts[:2]:
                if resid_h <= 0:
                    break
                j2 = eng.keys.index(s2)
                if any(jj == j2 and not (b1 < i0 or a1 > i1) for _, jj, a1, b1 in wins):   # reinforced itself at that time
                    continue
                spare_h = max(0.0, float((0.8 * cap[0, j2] - er_all[i0:i1 + 1, j2]).mean()) * 4)   # keep a 20 % margin
                if spare_h < 20:
                    continue
                if d <= a['walk_max_km']:
                    take = min(resid_h, spare_h)
                    measures.append({'mode': 'Walk', 'action': f"guide passengers on foot from {k} to {s2} ({'/'.join(_lines_of(eng, s2))}, {d} km)",
                                     'quantity': None, 'absorbs_per_hour': round(take),
                                     'basis': f'{s2} can take about {round(spare_h)} more passengers per hour'})
                else:
                    cycle = 2 * d / a['bus_speed_kmh'] * 60 + a['bus_loading_min']
                    per_bus = a['bus_capacity'] * 60 / cycle
                    want = min(resid_h, spare_h)
                    n = int(min(np.ceil(want / per_bus), a['buses_max']))
                    if n <= 0:
                        continue
                    take = min(want, n * per_bus)
                    measures.append({'mode': 'Bus', 'action': f"{n} shuttle bus(es) {k} -> {s2} ({'/'.join(_lines_of(eng, s2))}, {d} km, "
                                                              f"one every {cycle / max(n, 1):.0f} min)",
                                     'quantity': n, 'absorbs_per_hour': round(take),
                                     'basis': f"{a['bus_capacity']} passengers per bus, {cycle:.0f}-minute round trip; "
                                              f"{s2} can take about {round(spare_h)} more passengers per hour"})
                resid_h -= take
        order = ['taxi', 'bike'] if night else ['bike', 'taxi']
        for mode in order:
            if resid_h <= 0:
                break
            if mode == 'bike' and not night and rain <= a['bike_max_rain_mm_h']:
                n = int(min(np.ceil(resid_h / a['bike_pax_per_hour']), a['bikes_max']))
                if n <= 0:
                    continue
                take = min(resid_h, n * a['bike_pax_per_hour'])
                measures.append({'mode': 'Bike', 'action': f'bring {n} shared bikes to {k} (dry weather, short trips)', 'quantity': n,
                                 'absorbs_per_hour': round(take), 'basis': f"{a['bike_pax_per_hour']} passenger per bike per hour"})
                resid_h -= take
            if mode == 'taxi':
                n = int(min(np.ceil(resid_h / a['taxi_pax_per_hour']), a['taxis_max']))
                if n <= 0:
                    continue
                take = min(resid_h, n * a['taxi_pax_per_hour'])
                measures.append({'mode': 'Taxi', 'action': f'open a taxi rank at {k} for about {n} taxis', 'quantity': n,
                                 'absorbs_per_hour': round(take), 'basis': f"{a['taxi_pax_per_hour']} passengers per taxi per hour"})
                resid_h -= take
        if resid_h > 0:
            measures.append({'mode': 'Crowd control', 'action': f'meter access at the entrances of {k}: about {round(resid_h)} passengers per hour will have to wait',
                             'quantity': None, 'absorbs_per_hour': None, 'basis': 'what the other measures cannot absorb'})
        reason = _event_reason(eng, evs, k, idx[i0 + int(np.argmax(extra[i0:i1 + 1, j]))]) or \
            ('rain' if rain > 0.2 else 'demand above a normal day')
        by_mode = {}
        for m in measures:
            if m['absorbs_per_hour']:
                by_mode[m['mode']] = by_mode.get(m['mode'], 0) + m['absorbs_per_hour'] * hours
        extra_total = float(Ov.sum() - Oref.sum())
        plans.append({'station': k, 'lines': lines, 'window': f'{idx[i0]:%H:%M}-{idx[i1] + MIN15:%H:%M}', 'reason': reason,
                      'peak_demand_over_ceiling': round(float((L / c).max()), 2),
                      'extra_passengers_above_capacity_per_hour': round(extra_total / hours),
                      'measures': measures, 'hours': hours, 'extra_total': round(extra_total),
                      'absorbed_by_mode': {k2: round(min(v, extra_total)) for k2, v in by_mode.items()},
                      'residual_total': round(max(resid_h, 0) * hours)})
    line_summary = []
    for ln, lst in sorted(line_needs.items()):
        lst.sort()
        merged = [list(lst[0])]
        for s0, e0, df, f0 in lst[1:]:
            if s0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e0); merged[-1][2] = max(merged[-1][2], df)
            else:
                merged.append([s0, e0, df, f0])
        for s0, e0, df, f0 in merged:
            line_summary.append(f'{ln}: +{df} trains per hour {s0:%H:%M}-{e0:%H:%M} ({f0} -> {f0 + df})')
    summ = []
    if plans:
        summ.append('Extra U-Bahn trains: ' + ('; '.join(line_summary) if line_summary else 'not needed') + '.')
        for mode in ('S-Bahn', 'Walk', 'Bus', 'Bike', 'Taxi', 'Crowd control'):
            acts = [f"{p['station']} {p['window']}: {m['action']}" for p in plans for m in p['measures'] if m['mode'] == mode]
            if acts:
                summ.append(f'{mode}: ' + '; '.join(acts[:3]) + ('...' if len(acts) > 3 else '') + '.')
    else:
        summ.append('No station is expected to exceed what a normal dry day asks of it: no reinforcement needed.')
    return A._py({'date': day, 'weekday': day.day_name(), 'notes': notes, 'windows': plans, 'line_reinforcement': line_summary,
                  'summary': summ, 'assumptions': a,
                  'caution': 'the dataset only records U-Bahn station flows: S-Bahn, bus, taxi and bike quantities rest on the '
                             'planning assumptions listed here, and demand beyond the ceiling is estimated by the model'})
