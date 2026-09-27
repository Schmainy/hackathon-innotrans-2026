"""
ubahn_analyses.py - higher-level analysis tools built on UBahnEngine.

Each function reproduces one type of answer given during the training-question session, so an
agent calling them gets the same numbers. Every function takes the engine as first argument and
returns a JSON-serialisable dict with rounded numbers, sized to be read by a language model.

    event_report            what an event did (or will do) to the stations around its venue
    weather_peaks           the strongest bad-weather peak in a period, with time and stations
    closure_report          reason, duration, rerouting, overload and staffing for a closure
    peak_time               usual commute peak of a station vs the network
    energy_efficiency       energy per passenger by line, drivers and interventions
    fragmentation_ranking   stations whose closure splits the network the most
    anomaly_scan            unusual flows on a date and their most likely causes
    station_dependencies    non-adjacent station pairs whose demand moves together, and why
    disruption_response     what passengers actually do during closures and suspensions
    best_new_link           the short new link that most improves network resilience
    diversion_scenario      spreading an event crowd over an alternative, longer route
    route                   realistic route between two stations (changes penalised)
"""
from __future__ import annotations

import itertools
import re
from functools import lru_cache

import networkx as nx
import numpy as np
import pandas as pd

import json
import os
from src.engine import TOD_ORDER, UBahnEngine, _hav, _to_dt

MIN15 = pd.Timedelta('15min')


# ---------------------------------------------------------------------------------- helpers
def _py(x):
    """Recursively convert numpy / pandas values to plain Python for JSON."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else round(float(x), 3)
    if isinstance(x, float):
        return None if not np.isfinite(x) else round(x, 3)
    if isinstance(x, (pd.Timestamp,)):
        return x.strftime('%Y-%m-%d %H:%M')
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _reading(eng, lam: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(eng.expected_reading(lam.values, eng.ceiling[list(lam.columns)].values[None, :]),
                        index=lam.index, columns=lam.columns)


def _obs(eng, index, stations):
    return eng.flows.reindex(index)[stations]


def _window(eng, stations, start, end, events='data', closures='data', weather=None):
    """Observed vs normal (no events) vs model (with events) over a window, per station."""
    lam_n = eng.expected(start, end, stations, events=[], closures=closures, weather=weather)
    lam_m = eng.expected(start, end, stations, events=events, closures=closures, weather=weather)
    er_n, er_m = _reading(eng, lam_n), _reading(eng, lam_m)
    ob = _obs(eng, lam_n.index, stations)
    sd = np.sqrt((er_n ** 2).sum())
    return pd.DataFrame({'observed': ob.sum(), 'normal': er_n.sum().round(0), 'model': er_m.sum().round(0),
                         'excess_vs_normal': (ob.sum() - er_n.sum()).round(0),
                         'z': ((ob.sum() - er_n.sum()) / np.sqrt(sd ** 2 + 1)).round(1)})


@lru_cache(maxsize=4)
def _daily(eng):
    sd = eng.service_day(eng.flows.index)
    daily = eng.flows.groupby(sd).sum()
    dt = eng.daytype(daily.index)
    return daily, daily[dt == 0].mean(), daily[dt > 0].mean()


def _sbahn(eng, station):
    return bool(eng.stations.at[station, 'station_name'].startswith('S+U'))


def _weekday(ts):
    return pd.Timestamp(ts).day_name()


# ---------------------------------------------------------------------------------- routing
@lru_cache(maxsize=4)
def _line_graph(eng, transfer_penalty=3.0):
    """Line-expanded graph: nodes (station, line); riding = 1 per stop, changing line = penalty."""
    H = nx.Graph()
    for L, seq in eng.line_seq.items():
        for a, b in zip(seq[:-1], seq[1:]):
            H.add_edge((a, L), (b, L), weight=1.0, kind='ride')
    for s in eng.keys:
        ls = [L for L in eng.stations.at[s, 'lines'] if L in eng.line_seq]
        for l1, l2 in itertools.combinations(ls, 2):
            H.add_edge((s, l1), (s, l2), weight=transfer_penalty, kind='change')
    return H


def route(eng, origin, destination, avoid_line_segments=None, transfer_penalty=3.0, _with_rides=False):
    """Realistic route between two stations: fewest stops with each change of line counted as
    `transfer_penalty` stops. avoid_line_segments: list of (station_a, station_b, line) rides to forbid.

    Returns stops, changes and the legs, e.g. ['U2: Theodor-Heuss-Platz -> Alexanderplatz Bhf']."""
    a, b = eng.resolve_station(origin), eng.resolve_station(destination)
    H = _line_graph(eng, transfer_penalty).copy()
    for x, y, L in (avoid_line_segments or []):
        if H.has_edge((x, L), (y, L)):
            H.remove_edge((x, L), (y, L))
    H.add_node('SRC'); H.add_node('DST')
    for L in eng.stations.at[a, 'lines']:
        if (a, L) in H:
            H.add_edge('SRC', (a, L), weight=0.0)
    for L in eng.stations.at[b, 'lines']:
        if (b, L) in H:
            H.add_edge((b, L), 'DST', weight=0.0)
    try:
        p = nx.shortest_path(H, 'SRC', 'DST', weight='weight')[1:-1]
    except nx.NetworkXNoPath:
        return {'origin': a, 'destination': b, 'error': 'no U-Bahn route'}
    legs, rides = [], []
    for (s1, l1), (s2, l2) in zip(p[:-1], p[1:]):
        if l1 == l2:
            rides.append((s1, s2, l1))
            if legs and legs[-1][0] == l1:
                legs[-1][2] = s2
            else:
                legs.append([l1, s1, s2])
    stops = sum(1 for r in rides)
    out = {'origin': a, 'destination': b, 'stops': stops, 'changes': max(len(legs) - 1, 0),
           'legs': [f'{l}: {x} -> {y}' for l, x, y in legs]}
    if _with_rides:
        out['_rides'] = rides
    return out


# ---------------------------------------------------------------------------------- 1 events
def event_report(eng: UBahnEngine, query: str, date: str | None = None):
    """What an event did to the stations around its venue: arrival and departure waves, observed
    vs normal, ceiling hits, closures and weather that day, other events at the same venue.
    query matches event name, venue name, address or artist (case-insensitive)."""
    ev = eng.events
    q = query.lower()
    m = (ev.event_name.str.lower().str.contains(q, regex=False) | ev.venue_name.fillna('').str.lower().str.contains(q, regex=False)
         | ev.address.str.lower().str.contains(q, regex=False) | ev.venue_key.fillna('').str.lower().str.contains(q, regex=False))
    if not m.any():
        key = eng.geocode(query)[0]                       # e.g. 'Mercedes-Benz Arena' -> Uber Arena
        if key:
            m = ev.venue_key == key
    cand = ev[m].copy()
    if date:
        d = pd.Timestamp(date).normalize()
        cand = cand[eng.service_day(cand.start) == d]
    if cand.empty:
        return {'error': f'no event matching "{query}"' + (f' on {date}' if date else ''),
                'hint': 'the events file covers ' + f"{ev.start.min():%d %b} - {ev.start.max():%d %b %Y}"}
    groups = cand.groupby(['venue_key', 'start', 'end'], dropna=False)
    first_key = sorted(groups.groups.keys(), key=lambda k: k[1])[0]
    g = groups.get_group(first_key)
    vkey, s0, e0 = first_key
    att = float(g.estimated_attendance.sum())
    venue = g.address.iloc[0] + ' ' + str(g.venue_name.iloc[0] if isinstance(g.venue_name.iloc[0], str) else '')
    _, shares = eng.venue_shares(venue)
    stations = list(shares)
    day = eng.service_day(pd.DatetimeIndex([s0]))[0]
    in_data = s0 <= eng.flows.index.max()
    out = {'event': {'names': sorted(g.event_name.unique().tolist()), 'venue': vkey, 'address': g.address.iloc[0],
                     'start': s0, 'end': e0, 'end_is_dataset_estimate': True, 'weekday': _weekday(day),
                     'attendance_all_listings': att, 'listings': len(g)},
           'stations_and_shares': {k: round(v, 3) for k, v in shares.items()},
           'station_distance_km': {k: d for k, d in eng.stations_near(*eng.geocode(venue)[1:], 2.0) if k in shares},
           'other_matching_events': [_py({'name': r.event_name, 'start': r.start, 'attendance': r.estimated_attendance})
                                     for r in cand[(cand.start != s0)].itertuples()][:8]}
    if not in_data:
        out['note'] = 'event is after the data range: use event_impact for a forecast'
        return _py(out)
    wx = eng.weather.loc[day + pd.Timedelta('5h'):day + pd.Timedelta('1D 45min')]
    out['weather'] = {'daily_mean_temp': float(eng._daily_temp(pd.DatetimeIndex([day + pd.Timedelta('12h')]))[0]),
                      'rain_mm': float(wx.prcp.sum() / 4), 'max_rain_mm_h': float(wx.prcp.max())}
    out['closures_that_day'] = [f"{c['description']} ({c['start']:%H:%M}-{c['end']:%H:%M})" for c in eng._clist
                                if eng.service_day(pd.DatetimeIndex([c['start']]))[0] == day]
    out['other_events_that_day'] = [f"{e['name']} ({e['start']:%H:%M}, {int(e['attendance'])})"
                                    for e in eng._data_events(day + pd.Timedelta('5h'), day + pd.Timedelta('1D'))
                                    if e['start'] != s0]
    arr = _window(eng, stations, s0 - pd.Timedelta('90min'), s0 + pd.Timedelta('14min'))
    last = min(e0 + pd.Timedelta('1h59min'), eng.flows.index.max())
    dep = _window(eng, stations, e0, last)
    for name, w in (('arrival', arr), ('departure', dep)):
        tot = w.sum()
        out[f'{name}_window'] = {'from': (s0 - pd.Timedelta('90min')) if name == 'arrival' else e0,
                                 'to': (s0 + pd.Timedelta('14min')) if name == 'arrival' else last,
                                 'per_station': w.to_dict(orient='index'),
                                 'total_observed': tot.observed, 'total_normal': tot.normal, 'total_model': tot.model,
                                 'extra_passengers': tot.observed - tot.normal,
                                 'extra_as_share_of_attendance': (tot.observed - tot.normal) / max(att, 1)}
    idx = eng.slots(e0 - MIN15, last)
    top = list(dep.sort_values('excess_vs_normal', ascending=False).index[:3])
    lam_n = _reading(eng, eng.expected(idx[0], idx[-1], top, events=[]))
    out['departure_timeline'] = {s: [{'slot': t, 'observed': eng.flows.at[t, s], 'normal': round(lam_n.at[t, s])}
                                     for t in idx] for s in top}
    w0, w1 = s0 - pd.Timedelta('2h'), e0 + pd.Timedelta('2h')
    hits = {s: [t.strftime('%H:%M') for t in eng.flows.loc[w0:w1].index[eng.flows.loc[w0:w1, s] >= eng.ceiling[s]]]
            for s in stations}
    out['slots_at_ceiling'] = {s: h for s, h in hits.items() if h}
    out['ceilings'] = {s: eng.ceiling[s] for s in stations}
    # same venue on other dates, for comparison
    same = ev[(ev.venue_key == vkey) & (ev.start != s0) & (ev.start <= eng.flows.index.max())]
    comp = []
    for s1, gg in same.groupby('start'):
        e1 = gg.end.max()
        w = _window(eng, stations, e1, min(e1 + pd.Timedelta('1h59min'), eng.flows.index.max()))
        comp.append({'date': s1, 'attendance': gg.estimated_attendance.sum(), 'departure_extra': w.excess_vs_normal.sum(),
                     'peak_reading_after': int(eng.flows.loc[e1:e1 + pd.Timedelta('2h'), stations].max().max())})
    out['same_venue_other_dates'] = comp[:10]
    # plain-language facts from the recorded data, listed first so they lead the answer
    d, a = out['departure_window'], out['arrival_window']
    top_st = top[0] if top else None
    facts = [f"Recorded data (not a forecast): {', '.join(g.event_name.unique()[:1])} at {vkey}, {_weekday(day)} {s0:%d %b}, "
             f"{s0:%H:%M}-{e0:%H:%M} (end time is the dataset's estimate), {int(att)} attendees."]
    facts.append(f"Departure wave {e0:%H:%M}-{min(e0 + pd.Timedelta('2h'), last + MIN15):%H:%M}: {int(d['total_observed'])} passengers at {', '.join(stations)} "
                 f"against about {int(d['total_normal'])} normally ({int(d['extra_passengers']):+d}, "
                 f"{d['extra_as_share_of_attendance']:.0%} of attendance).")
    if top_st:
        tl = out['departure_timeline'][top_st]
        best = max(tl, key=lambda x: x['observed'] - x['normal'])
        facts.append(f"Busiest slot after the show: {top_st} {int(best['observed'])} at {best['slot']:%H:%M} (normal about {best['normal']}).")
    facts.append(f"Arrival window {s0 - pd.Timedelta('90min'):%H:%M}-{s0 + MIN15:%H:%M}: {int(a['total_observed'])} against about "
                 f"{int(a['total_normal'])} normally ({a['total_observed'] / max(a['total_normal'], 1) - 1:+.0%})"
                 + ("; the arrivals overlap the evening rush, so they barely stand out in single readings." if 16 <= s0.hour <= 19 else "."))
    facts.append('Ceiling reached: ' + ('; '.join(f"{k} at {', '.join(v)}" for k, v in out['slots_at_ceiling'].items())
                                         if out['slots_at_ceiling'] else 'no station around the venue reached its ceiling in that period') + '.')
    facts.append('Closures that day: ' + ('; '.join(out['closures_that_day']) if out['closures_that_day'] else 'none') + '.')
    if comp:
        bigc = max(comp, key=lambda c: c['attendance'])
        facts.append(f"Largest other show at this venue: {bigc['date']:%d %b} ({int(bigc['attendance'])} attendees): "
                     f"departure extra {int(bigc['departure_extra']):+d}, highest reading after it {bigc['peak_reading_after']}.")
    out = {'facts_from_recorded_data': facts, **out}
    return _py(out)


# ---------------------------------------------------------------------------------- 2 weather
def weather_peaks(eng: UBahnEngine, start: str, end: str):
    """Strongest bad-weather peak between two dates: rain event, network peak slot vs a dry day of the
    same weekday, stations at their ceiling, top station readings, and how much the rain rule explains."""
    s, e = pd.Timestamp(start).normalize() + pd.Timedelta('5h'), pd.Timestamp(end).normalize() + pd.Timedelta('1D 45min')
    wx = eng.weather.loc[s:e]
    sd = eng.service_day(wx.index)
    days = wx.groupby(sd).agg(rain_mm=('prcp', lambda x: x.sum() / 4), max_rain_mm_h=('prcp', 'max'),
                              temp_mean=('temp', 'mean'), wind_max=('wspd', 'max'))
    days['weekday'] = [d.day_name() for d in days.index]
    t_peak_rain = wx.prcp.idxmax()
    if wx.prcp.max() <= 0:
        return {'daily_weather': _py(days.round(1).reset_index().to_dict(orient='records')), 'note': 'no rain in the period'}
    day = eng.service_day(pd.DatetimeIndex([t_peak_rain]))[0]
    dayw = eng.weather.loc[day + pd.Timedelta('5h'):day + pd.Timedelta('1D 45min')]
    wet = dayw.prcp > 0
    # contiguous rain window around the peak
    i = dayw.index.get_loc(t_peak_rain); a = b = i
    while a > 0 and wet.iloc[a - 1]:
        a -= 1
    while b < len(dayw) - 1 and wet.iloc[b + 1]:
        b += 1
    r0, r1 = dayw.index[a], dayw.index[b]
    F = eng.flows
    net = F.sum(axis=1)
    same = [d for d in pd.date_range(F.index.min().normalize(), F.index.max().normalize())
            if d.dayofweek == day.dayofweek and d != day]
    daily_rain = eng.weather.prcp.groupby(eng.service_day(eng.weather.index)).sum() / 4
    dry = [d for d in same if daily_rain.get(d, 1) < 0.5]
    tod = pd.Index(net.loc[r0:r1].index.strftime('%H:%M'))
    typical = pd.DataFrame({d: net.loc[d + (r0 - day):d + (r1 - day)].values for d in dry if len(net.loc[d + (r0 - day):d + (r1 - day)]) == len(tod)}).median(axis=1)
    typical.index = net.loc[r0:r1].index
    ratio = net.loc[r0:r1] / typical
    tpk = net.loc[r0:r1].idxmax()
    rank = int((net > net[tpk]).sum()) + 1
    at_ceiling = int((F.loc[tpk] >= eng.ceiling).sum())
    at_ceiling_dry = [int((F.loc[d + (tpk - day)] >= eng.ceiling).sum()) for d in dry]
    tday = float(eng._daily_temp(pd.DatetimeIndex([day + pd.Timedelta('12h')]))[0])
    lam_dry = eng.expected(tpk, tpk, weather={'tmean': tday, 'prcp': 0})
    er_dry = _reading(eng, lam_dry).iloc[0]
    top = F.loc[tpk].sort_values(ascending=False).head(6)
    top_readings = [{'station': k, 'observed': v, 'expected_without_rain': round(er_dry[k]), 'ceiling': eng.ceiling[k],
                     'at_ceiling': bool(v >= eng.ceiling[k])} for k, v in top.items()]
    win = F.loc[r0:r1]
    st_max = win.max().sort_values(ascending=False).head(5)
    obs_tot = win.values.sum()
    no_rain = _reading(eng, eng.expected(r0, r1, weather={'tmean': tday, 'prcp': 0})).values.sum()
    with_rain = _reading(eng, eng.expected(r0, r1)).values.sum()
    out = {'daily_weather': days.round(1).reset_index().rename(columns={'index': 'day'}).to_dict(orient='records'),
           'strongest_rain': {'day': day, 'weekday': day.day_name(), 'rain_window': f'{r0:%H:%M}-{r1:%H:%M}',
                              'peak_intensity_mm_h': float(wx.prcp.max()), 'peak_rain_at': t_peak_rain,
                              'rain_total_mm': float(dayw.prcp.sum() / 4),
                              'temp_range_in_window': [float(dayw.loc[r0:r1].temp.min()), float(dayw.loc[r0:r1].temp.max())],
                              'wind_max_kmh': float(dayw.loc[r0:r1].wspd.max()),
                              'is_most_intense_rain_in_dataset': bool(eng.weather.prcp.max() == wx.prcp.max())},
           'network_peak': {'slot': tpk, 'passengers_15min': net[tpk], 'typical_dry_same_weekday': round(typical[tpk]),
                            'ratio': ratio[tpk], 'rank_among_all_slots': rank, 'of_slots': len(net),
                            'stations_at_ceiling': at_ceiling,
                            'stations_at_ceiling_dry_same_weekdays': [min(at_ceiling_dry), max(at_ceiling_dry)] if at_ceiling_dry else None},
           'network_ratio_by_slot': {t.strftime('%H:%M'): round(v, 2) for t, v in ratio.items()},
           'top_station_readings_at_peak': top_readings,
           'highest_station_readings_in_rain_window': [{'station': k, 'reading': v, 'at': win[k].idxmax(), 'ceiling': eng.ceiling[k]}
                                                        for k, v in st_max.items()],
           'rain_window_totals': {'observed': obs_tot, 'expected_without_rain': round(no_rain), 'expected_with_rain': round(with_rain),
                                  'excess_vs_no_rain': obs_tot / no_rain - 1, 'model_gap': obs_tot / with_rain - 1},
           'events_in_window': [e['name'] for e in eng._data_events(r0, r1)],
           'closures_in_window': [c['description'] for c in eng._clist if c['start'] <= r1 and c['end'] >= r0],
           'noise_note': 'single stations reach their ceiling 50-270 times per summer, dry days included; '
                         'the network totals are the solid evidence'}
    return _py(out)


# ---------------------------------------------------------------------------------- 3 closures
def closure_report(eng: UBahnEngine, line: str | None = None, station_a: str | None = None,
                   station_b: str | None = None, station: str | None = None, date: str | None = None):
    """Look up a closure in the data (by line + section stations, or by station, and/or date) and report
    reason, duration, rerouting, what the data shows, overload risk (actual time and evening peak) and
    where to put staff."""
    cands = []
    for c in eng._clist:
        if date and eng.service_day(pd.DatetimeIndex([c['start']]))[0] != pd.Timestamp(date).normalize():
            continue
        if line and c['line'] != line.upper():
            continue
        names = [s for s in (station_a, station_b, station) if s]
        ok = True
        for s in names:
            try:
                ks = eng.resolve_station(s, multi=True)
            except ValueError:
                ok = False; break
            if c['kind'] == 'line':
                if not any(k in (c['stations'][0], c['stations'][-1]) or k in c['stations'] for k in ks):
                    ok = False
            elif not set(ks) & set(c['stations']):
                ok = False
        if ok:
            cands.append(c)
    if not cands:
        return {'error': 'no matching closure in the data', 'closures_available': len(eng._clist)}
    exact = [c for c in cands if c['kind'] == 'line' and station_a and station_b and
             {c['stations'][0], c['stations'][-1]} == {eng.resolve_station(station_a), eng.resolve_station(station_b)}]
    c = (exact or cands)[0]
    reason = re.search(r'due to (.+?)\.?$', c['description'])
    dur = c['end'] - c['start']
    out = {'closure': {'description': c['description'], 'reason': reason.group(1) if reason else None,
                       'start': c['start'], 'end': c['end'], 'weekday': _weekday(c['start']),
                       'duration': f'{int(dur.total_seconds() // 3600)} h {int(dur.total_seconds() % 3600 // 60):02d} min'},
           'other_matches': [f"{x['description']} ({x['start']:%d %b %H:%M})" for x in cands if x is not c]}
    if c['kind'] == 'line':
        L, sec = c['line'], c['stations']
        ro = eng.reroute_options(L, sec[0], sec[-1])
        out['rerouting'] = {'section': ro['section'], 'stations_without_this_line': ro['interior_without_service'],
                            'line_split_into': ro['line_split_into'],
                            'u_bahn_detour_between_ends': ro['detour_by_line'],
                            'per_station': {s: {'other_lines': v['other_lines_here'], 's_bahn_here': _sbahn(eng, s),
                                                'walk_to': v['walkable_alternatives_km'][:3]}
                                            for s, v in ro['per_station_alternatives'].items()}}
        alts = sorted({k for s in ro['interior_without_service'] for k, _ in ro['per_station_alternatives'][s]['walkable_alternatives_km']} - set(ro['section']))
        S, E = c['start'].ceil('15min'), c['end'] - pd.Timedelta('1min')
        data = {}
        for name, sts in (('interior', ro['interior_without_service']), ('section_ends', [sec[0], sec[-1]]), ('walkable_alternatives', alts)):
            if sts:
                w = _window(eng, sts, S, E).sum()
                data[name] = {'stations': sts, 'observed_total_over_the_closure': w.observed,
                              'expected_total_over_the_closure': w.model, 'change_pct': (w.observed / max(w.model, 1) - 1) * 100}
        out['what_the_data_shows'] = {'window': f"{S:%H:%M}-{c['end']:%H:%M}",
                                      'note': 'totals over the whole closure window, summed over each group of stations (not per 15 minutes)', **data}
        minutes = int(max(dur.total_seconds() // 60, 15))
        sc_now = eng.suspension_scenario(L, sec[0], sec[-1], c['start'].floor('15min'), minutes=minutes)
        peak = c['start'].normalize() + pd.Timedelta('17h30min')
        sc_peak = eng.suspension_scenario(L, sec[0], sec[-1], peak, minutes=90)
        def fmt(df):
            return [{'station': k, 'mean_passengers_15min': round(r.lambda_mean), 'ceiling': r.ceiling,
                     'chance_at_ceiling_any_slot': r.p_any_slot} for k, r in df.head(4).iterrows()]
        now = fmt(sc_now['planning_case'])
        worst_now = max([r_['chance_at_ceiling_any_slot'] for r_ in now] or [0])
        out['overload'] = {'at_the_recorded_time': {'window': f"{c['start']:%a %d %b %H:%M}-{c['end']:%H:%M}", 'stations': now,
                                                    'summary': ('no station near its ceiling during the recorded window'
                                                                if worst_now < 0.2 else 'stations at risk during the recorded window')},
                           'hypothetical_if_it_happened_at_the_17_30_peak': {'label': 'hypothetical case, not what happened',
                                                                            'stations': fmt(sc_peak['planning_case'])},
                           'hypothetical_17_30_without_rerouting': fmt(sc_peak['as_in_data']),
                           'no_u_bahn_alternative': sc_now['no_u_bahn_alternative'],
                           'note': 'planning case = displaced passengers walk to the nearest station with another line; '
                                   'the recorded data shows no rerouting at all'}
        out['section_ends'] = [{'station': s_, 'lines': eng.stations.at[s_, 'lines'], 'interchange': len(eng.stations.at[s_, 'lines']) > 1}
                               for s_ in (sec[0], sec[-1])]
        staff = [sec[0], sec[-1]] + [s for s in ro['interior_without_service'] if ro['per_station_alternatives'][s]['other_lines_here']]
        staff += [s for s in ro['interior_without_service'] if _sbahn(eng, s)]
        out['staff_at'] = list(dict.fromkeys(staff))
        out['staff_note'] = ('for the recorded window: at the section ends and transfer points to guide passengers'
                             + (', a light presence is enough since no station was near its ceiling' if worst_now < 0.2 else ''))
    else:
        k = c['stations'][0]
        S, E = c['start'].ceil('15min'), c['end'] - pd.Timedelta('1min')
        lost = _reading(eng, eng.expected(S, E, [k], closures=[])).values.sum()
        nb = list(eng.graph.neighbors(k))
        w = _window(eng, nb, S, E).sum()
        t = c['end'].ceil('15min')
        after = eng.slots(t, t + pd.Timedelta('30min'))
        normal_after = _reading(eng, eng.expected(after[0], after[-1], [k], closures=[]))[k]
        out['station'] = {'name': k, 'passengers_lost_during_closure': round(lost), 's_bahn_here': _sbahn(eng, k),
                          'adjacent_stations': nb, 'adjacent_observed': w.observed, 'adjacent_expected': w.model,
                          'after_reopening': [{'slot': x, 'observed': eng.flows.at[x, k] if x in eng.flows.index else None,
                                               'normal': round(normal_after[x])} for x in after],
                          'rebound_rule': '39 %, 12 %, 5 % of the lost volume in the first three slots after reopening'}
        out['staff_at'] = [k] + nb
    return _py(out)


# ---------------------------------------------------------------------------------- 4 peak time
def peak_time(eng: UBahnEngine, station: str):
    """Usual weekday commute peak of a station (time and level), compared with all stations."""
    k = eng.resolve_station(station)
    F = eng.flows
    dt = eng.daytype(eng.service_day(F.index))
    res = {}
    for name, sel, agg in (('weekday_mean', dt == 0, 'mean'), ('all_days_mean', slice(None), 'mean'), ('weekday_median', dt == 0, 'median')):
        sub = F[sel]
        prof = sub.groupby(sub.index.strftime('%H:%M')).agg(agg).reindex(TOD_ORDER)
        r = prof[k]
        res[name] = {'peak_time': r.idxmax(), 'peak_value': r.max(),
                     'morning_peak': [r.loc['06:00':'10:00'].idxmax(), r.loc['06:00':'10:00'].max()],
                     'evening_peak': [r.loc['15:00':'20:00'].idxmax(), r.loc['15:00':'20:00'].max()],
                     'mean_of_all_station_peaks': prof.max().mean(), 'median_of_all_station_peaks': prof.max().median(),
                     'rank': int((prof.max() > r.max()).sum()) + 1}
        if name == 'weekday_mean':
            res[name]['network_average_at_evening_peak_slot'] = prof.loc[r.loc['15:00':'20:00'].idxmax()].mean()
            res[name]['network_average_at_morning_peak_slot'] = prof.loc[r.loc['06:00':'10:00'].idxmax()].mean()
    wd = F[dt == 0]
    h = wd.index.strftime('%H:%M')
    days = eng.service_day(wd.index)
    am = wd[(h >= '07:00') & (h <= '08:45')].groupby(days[(h >= '07:00') & (h <= '08:45')]).mean()
    pm = wd[(h >= '17:00') & (h <= '18:45')].groupby(days[(h >= '17:00') & (h <= '18:45')]).mean()
    rng = np.random.default_rng(0)
    boot = [pm[k].values[s].mean() / am[k].values[s].mean() for s in (rng.choice(len(am), len(am)) for _ in range(1000))]
    ceiling_days = wd[k].groupby(days).max() >= eng.ceiling[k]
    wdm = res['weekday_mean']
    diff = (wdm['peak_value'] / wdm['mean_of_all_station_peaks'] - 1) * 100
    # the same comparison on the data held before the last injection (e.g. the training period), when there was one
    before = None
    lr_path = os.path.join(eng.data_dir, 'learning_report.json')
    if os.path.exists(lr_path):
        try:
            with open(lr_path, encoding='utf-8') as fh:
                end_b = pd.Timestamp(json.load(fh)['period_before'][1])
            Fb = F[F.index <= end_b]
            dtb = eng.daytype(eng.service_day(Fb.index))
            sub = Fb[dtb == 0]
            prof_b = sub.groupby(sub.index.strftime('%H:%M')).mean().reindex(TOD_ORDER)
            pk, mean_b = prof_b[k].max(), prof_b.max().mean()
            d_b = (pk / mean_b - 1) * 100
            days_b = sorted(set(eng.service_day(sub.index)))
            before = {'period': f"{days_b[0]:%Y-%m-%d} -> {days_b[-1]:%Y-%m-%d}, {len(days_b)} weekdays (data before the last injection)",
                      'peak_time': prof_b[k].idxmax(), 'station_peak': pk, 'mean_of_all_station_peaks': mean_b,
                      'difference_pct': round(d_b, 2),
                      'verdict': 'about equal (within 2 %)' if abs(d_b) < 2 else ('above' if d_b > 0 else 'below')}
        except Exception:
            before = None
    wdays = sorted(set(eng.service_day(F.index[dt == 0])))
    out = {'station': k, 'lines': eng.stations.at[k, 'lines'], 'ceiling': eng.ceiling[k],
           'period_used': f"{wdays[0]:%Y-%m-%d} -> {wdays[-1]:%Y-%m-%d}, {len(wdays)} weekdays (all recorded data)",
           'comparison_with_mean_of_all_station_peaks': {'station_peak': wdm['peak_value'], 'mean_of_all_station_peaks': wdm['mean_of_all_station_peaks'],
                                                        'difference_pct': round(diff, 2),
                                                        'verdict': 'about equal (within 2 %)' if abs(diff) < 2 else ('above' if diff > 0 else 'below')},
           **({'comparison_before_last_injection': before} if before else {}),
           'definitions': res,
           'evening_vs_morning_ratio': {'station': pm[k].mean() / am[k].mean(), 'ci95': list(np.percentile(boot, [2.5, 97.5])),
                                       'network': pm.mean(axis=1).mean() / am.mean(axis=1).mean()},
           'model_shared_profile_peaks': ['08:00', '18:00'],
           'share_of_weekdays_hitting_ceiling': ceiling_days.mean(),
           'south_west_cluster_note': 'the mean over stations is pulled up by Spichernstr., Kurfürstendamm, Berliner Str., '
                                      'Wittenbergplatz and neighbours (2-4x a typical station)'}
    return _py(out)


# ---------------------------------------------------------------------------------- 5 energy
def energy_efficiency(eng: UBahnEngine):
    """Energy per passenger by line (interchange stations counted for each line they serve), fixed vs
    per-passenger energy, demand density, weekday/weekend, and interventions for the worst line."""
    daily, wd, we = _daily(eng)
    E = eng.energy.astype(float).copy(); E.index = pd.to_datetime(E.index).normalize()
    dt = pd.Series(eng.daytype(daily.index), index=daily.index)
    nl = eng.stations['lines'].map(len)
    rows, series = {}, {}
    for L in E.columns:
        ks = [k for k in eng.keys if L in eng.stations.at[k, 'lines']]
        R = daily[ks].sum(axis=1); e = E[L].reindex(daily.index); ok = e.notna()
        R, e, d = R[ok], e[ok], dt[ok]
        b1, b0 = np.polyfit(R, e, 1)
        seq = eng.line_seq[L]
        length = float(sum(_hav(eng.stations.at[a, 'latitude'], eng.stations.at[a, 'longitude'],
                                eng.stations.at[b, 'latitude'], eng.stations.at[b, 'longitude']) for a, b in zip(seq[:-1], seq[1:])))
        split = (daily[ks] / nl[ks].values).sum(axis=1)[ok]
        rows[L] = {'kwh_per_passenger': e.sum() * 1000 / R.sum(), 'kwh_per_passenger_if_interchange_passengers_split_between_lines': e.sum() * 1000 / split.sum(),
                   'energy_mwh_day': e.mean(), 'passengers_day': R.mean(), 'length_km': length, 'stations': len(ks),
                   'passengers_per_km': R.mean() / length, 'fixed_share': b0 / e.mean(), 'fixed_mwh_day': b0,
                   'marginal_wh_per_passenger': b1 * 1e6, 'mwh_per_km': e.mean() / length,
                   'kwh_per_passenger_weekday': e[d == 0].sum() * 1000 / R[d == 0].sum(),
                   'kwh_per_passenger_weekend': e[d > 0].sum() * 1000 / R[d > 0].sum()}
        series[L] = (R, e, d, b0, b1)
    df = pd.DataFrame(rows).T.sort_values('kwh_per_passenger', ascending=False)
    worst = df.index[0]
    R, e, d, b0, b1 = series[worst]
    base = e.sum() / R.sum()
    rr = R[d > 0].mean() / R[d == 0].mean()
    e_we = e.copy(); e_we[d > 0] = e[d > 0] - b0 * (1 - rr)
    tot_e = sum(series[L][1].sum() for L in series); tot_r = sum(series[L][0].sum() for L in series)
    med_x = df.passengers_per_km.median() * df.at[worst, 'length_km'] / R.mean() - 1
    e_med = e + b1 * R * med_x
    out = {'counting': 'default: passengers per line = daily flow of every station the line serves, so an interchange station counts '
                       'fully for each of its lines (this matches the energy data best). Alternative: an interchange station\'s flow '
                       'is split equally between its lines.',
           'by_line': df.round(3).to_dict(orient='index'),
           'worst_line': worst, 'best_lines': list(df.index[-3:][::-1]), 'network_kwh_per_passenger': tot_e * 1000 / tot_r,
           'worst_line_if_interchange_passengers_split_between_lines': df['kwh_per_passenger_if_interchange_passengers_split_between_lines'].idxmax(),
           'correlations': {'kwh_per_passenger_vs_passengers_per_km': np.corrcoef(df.kwh_per_passenger.astype(float), df.passengers_per_km.astype(float))[0, 1],
                            'fixed_energy_vs_length': np.corrcoef(df.fixed_mwh_day.astype(float), df.length_km.astype(float))[0, 1]},
           'weather_effect_on_energy': 'none beyond ridership (residual correlation with temperature |r| <= 0.12)',
           'interventions_nature': 'hypothetical scenarios computed with the fitted energy model (fixed energy plus energy per passenger), '
                                   'not measured effects of a timetable change',
           'interventions_for_worst_line': {
               'line': worst, 'overall_kwh_per_passenger_before': base * 1000,
               'cut_fixed_energy_10pct': {'overall_kwh_per_passenger_after': (e - 0.1 * b0).sum() / R.sum() * 1000,
                                          'change_pct': ((e - 0.1 * b0).sum() / R.sum() / base - 1) * 100, 'mwh_saved_per_day': 0.1 * b0},
               'ridership_plus_10pct': {'overall_kwh_per_passenger_after': (e + b1 * R * 0.1).sum() / (R * 1.1).sum() * 1000,
                                        'change_pct': ((e + b1 * R * 0.1).sum() / (R * 1.1).sum() / base - 1) * 100,
                                        'extra_mwh_per_day': b1 * R.mean() * 0.1},
               'weekend_service_matched_to_demand': {'weekend_ridership_as_share_of_weekday_pct': rr * 100,
                                                     'weekend_only_kwh_per_passenger_before': e[d > 0].sum() / R[d > 0].sum() * 1000,
                                                     'overall_kwh_per_passenger_after': e_we.sum() / R.sum() * 1000,
                                                     'change_pct': (e_we.sum() / R.sum() / base - 1) * 100,
                                                     'mwh_saved_per_weekend_day': b0 * (1 - rr)},
               'reach_median_line_density': {'ridership_increase_needed_pct': med_x * 100,
                                             'overall_kwh_per_passenger_after': e_med.sum() / (R * (1 + med_x)).sum() * 1000}}}
    return _py(out)


# ---------------------------------------------------------------------------------- 6 fragmentation
def fragmentation_ranking(eng: UBahnEngine, top: int = 5):
    """Stations whose closure (station and tracks) splits the network the most, with passengers affected
    (own weekday flow + flow at stations cut off, an upper bound) and walking links across the gap."""
    G = eng.graph; n = G.number_of_nodes(); pairs_all = (n - 1) * (n - 2) // 2
    _, wd, we = _daily(eng)
    eff0 = nx.global_efficiency(G)
    rows = []
    for v in G.nodes():
        H = G.copy(); H.remove_node(v)
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        lost = (n - 1) * (n - 2) // 2 - sum(len(c) * (len(c) - 1) // 2 for c in comps)
        rows.append((v, lost, comps, 1 - nx.global_efficiency(H) / eff0))
    rows.sort(key=lambda r: (-r[1], -r[3]))
    out, seen = [], []
    for v, lost, comps, effd in rows[:top + 3]:
        main = comps[0]
        pieces = []
        for c in comps[1:]:
            c = list(c)
            nb = [x for x in c if G.has_edge(x, v)]
            far = max(c, key=lambda x: nx.shortest_path_length(G, v, x))
            lines = sorted(set(sum([eng.stations.at[x, 'lines'] for x in c], [])))
            links = []
            for x in c:
                for k, dd in eng.stations_near(eng.stations.at[x, 'latitude'], eng.stations.at[x, 'longitude'], 1.5):
                    if k in main:
                        links.append((dd, x, k)); break
            links.sort()
            pieces.append({'stations': len(c), 'lines': lines, 'from': nb[0] if nb else c[0], 'to': far,
                           'weekday_passengers': wd[c].sum(), 'walk_links': [f'{x} -> {k} ({dd} km)' for dd, x, k in links[:2]],
                           's_bahn_stations_in_piece': [x for x in c if _sbahn(eng, x)]})
        cut = set().union(*comps[1:]) if len(comps) > 1 else set()
        out.append({'station': v, 'lines': eng.stations.at[v, 'lines'], 'station_pairs_disconnected_share': lost / pairs_all,
                    'stations_cut_off': len(cut), 'efficiency_loss': effd, 'pieces': pieces,
                    'passengers_affected_weekday': wd[v] + (wd[list(cut)].sum() if cut else 0),
                    'passengers_affected_weekend': we[v] + (we[list(cut)].sum() if cut else 0),
                    'same_branch_as_a_higher_rank': any(len(cut & s) / max(len(cut | s), 1) > 0.7 for s in seen)})
        seen.append(cut)
    ranked = out[:top]
    extra = [o for o in out[top:] if not o['same_branch_as_a_higher_rank']][:1]
    return _py({'method': 'remove each station and its tracks; count station pairs that can no longer reach each other',
                'condition': 'assumes the station and its tracks are unusable, so trains cannot pass; if trains can pass through, '
                             'only the station\'s own passengers are affected',
                'ranking': ranked, 'next_distinct_weak_point': extra,
                'mitigation_principles': ['keep trains running through the closed station when tracks are usable '
                                          '(then only its own passengers are affected - as in all 11 closures in the data)',
                                          'if tracks are out: shuttle trains on cut-off branches + replacement buses across the gap',
                                          'plan works at quiet times: midday lull, nights, weekends, never on event days']})


# ---------------------------------------------------------------------------------- 7 anomalies
def anomaly_scan(eng: UBahnEngine, date: str, top: int = 8):
    """Unusual flows on one service day and their most likely causes: event clusters, weather, hidden
    closures, isolated spikes (with the normal spike rate for comparison) and energy."""
    day = pd.Timestamp(date).normalize()
    S, E = day + pd.Timedelta('5h'), day + pd.Timedelta('1D 45min')
    obs = eng.flows.loc[S:E]
    base = eng.expected(S, E, events=[], closures='data'); full = eng.expected(S, E)
    erb, erf = _reading(eng, base), _reading(eng, full)
    o4, b4, f4 = obs.rolling(4).sum(), erb.rolling(4).sum(), erf.rolling(4).sum()
    zb = (o4 - b4) / np.sqrt((erb ** 2).rolling(4).sum() + 1); zf = (o4 - f4) / np.sqrt((erf ** 2).rolling(4).sum() + 1)
    stk = pd.DataFrame({'z_vs_normal': zb.stack(), 'z_vs_model': zf.stack(), 'observed': o4.stack(), 'normal': b4.stack(), 'model': f4.stack()}).reset_index()
    stk.columns = ['window_end', 'station', 'z_vs_normal', 'z_vs_model', 'observed', 'normal', 'model']
    stk['window'] = [f'{(t - pd.Timedelta("45min")):%H:%M}-{(t + MIN15):%H:%M}' for t in stk.window_end]
    best = stk.sort_values('z_vs_normal', ascending=False).drop_duplicates('station').head(top)
    events = []
    for e in eng._data_events(S, E):
        key, sh = eng.venue_shares(e['venue']); st = list(sh)
        for label, a, b in (('arrival', e['start'] - pd.Timedelta('90min'), e['start'] + pd.Timedelta('14min')),
                            ('departure', e['end'], min(e['end'] + pd.Timedelta('1h59min'), E))):
            w = _window(eng, st, a, b).sum()
            sdv = np.sqrt((_reading(eng, eng.expected(a, b, st, events=[])) ** 2).values.sum())
            events.append({'event': e['name'], 'venue': key, 'attendance': e['attendance'], 'phase': label,
                           'window': f'{a:%H:%M}-{b:%H:%M}', 'stations': st, 'observed': w.observed, 'normal': w.normal,
                           'model_with_event': w.model, 'extra': w.observed - w.normal,
                           'extra_share_of_attendance': (w.observed - w.normal) / max(e['attendance'], 1),
                           'z': (w.observed - w.normal) / max(sdv, 1)})
    # possible spill-over: strong excess at a station outside every event's split, attributed to the nearest venue
    extra_near = []
    day_events = []
    for e in eng._data_events(S, E):
        _, lat, lon = eng.geocode(e['venue'])
        if np.isfinite(lat):
            day_events.append((e, lat, lon, eng.venue_shares(e['venue'])[1]))
    in_any_split = set().union(*[set(v[3]) for v in day_events]) if day_events else set()
    for e, lat, lon, _ in day_events:
        a, b = e['end'], min(e['end'] + pd.Timedelta('1h45min'), E)
        o = eng.flows.loc[a:b]; bb = _reading(eng, eng.expected(a, b, events=[]))
        ex = o.sum() - bb.sum(); z = ex / np.sqrt((bb ** 2).sum() + 1)
        for k, dkm in eng.stations_near(lat, lon, 2.5):
            if k in in_any_split or z[k] <= 5:
                continue
            nearest = min(day_events, key=lambda v: _hav(v[1], v[2], eng.stations.at[k, 'latitude'], eng.stations.at[k, 'longitude']))
            if nearest[0] is e:
                extra_near.append({'event': e['name'], 'station': k, 'km': dkm, 'extra': ex[k], 'z': z[k],
                                   'note': 'possible spill-over, weaker evidence than the event split'})
    net = obs.sum(axis=1) / erb.sum(axis=1)
    tday = float(eng._daily_temp(pd.DatetimeIndex([day + pd.Timedelta('12h')]))[0])
    wx = eng.weather.loc[S:E]
    rain = wx[wx.prcp > 0]
    weather = {'daily_mean_temp': tday, 'rain_mm': wx.prcp.sum() / 4, 'max_rain_mm_h': wx.prcp.max(), 'wind_max_kmh': wx.wspd.max(),
               'rain_slots': f'{rain.index.min():%H:%M}-{rain.index.max():%H:%M}' if len(rain) else None,
               'hourly_temps': wx[wx.index.minute == 0].temp.round(1).tolist()}
    episodes = []
    if len(rain):
        # contiguous rain episodes (gaps up to 30 min merged)
        times = list(rain.index); grp = [[times[0]]]
        for t in times[1:]:
            (grp[-1].append(t) if t - grp[-1][-1] <= pd.Timedelta('30min') else grp.append([t]))
        same = [d for d in pd.date_range(eng.flows.index.min().normalize(), eng.flows.index.max().normalize()) if d.dayofweek == day.dayofweek]
        for g_ in grp:
            r0, r1 = g_[0].floor('h'), g_[-1]
            if r1 - r0 < pd.Timedelta('30min') or r1 > eng.flows.index.max():
                continue
            o = eng.flows.loc[r0:r1].values.sum()
            dry = _reading(eng, eng.expected(r0, r1, weather={'tmean': tday, 'prcp': 0})).values.sum()
            wet = _reading(eng, eng.expected(r0, r1)).values.sum()
            ratios = []
            for d in same:
                a, b = d + (r0 - day), d + (r1 - day)
                if b <= eng.flows.index.max() and d != day:
                    ratios.append(eng.flows.loc[a:b].values.sum() / _reading(eng, eng.expected(a, b)).values.sum())
            episodes.append({'window': f'{r0:%H:%M}-{r1 + MIN15:%H:%M}', 'max_rain_mm_h': float(eng.weather.prcp.loc[g_[0]:g_[-1]].max()),
                             'temp_drop': float(eng.weather.temp.loc[r0 - pd.Timedelta('1h'):r1].max() - eng.weather.temp.loc[r0:r1].min()),
                             'wind_max_kmh': float(eng.weather.wspd.loc[r0:r1].max()),
                             'observed': o, 'expected_dry': round(dry), 'expected_with_rain': round(wet),
                             'excess_vs_dry': o / dry - 1, 'obs_over_model': o / wet,
                             'share_of_excess_explained_by_rain': (wet - dry) / (o - dry) if o > dry else None,
                             'usual_week_to_week_sd_of_obs_over_model': float(np.std(ratios)) if ratios else None})
    weather['rain_episodes'] = episodes
    # spike baseline
    t0, t1 = eng.flows.index.min(), eng.flows.index.max()
    L = eng.expected(t0, t1).values; O = eng.flows.reindex(eng.slots(t0, t1)).values
    spike = (O > 8 * L) & (L > 3)
    per_day = pd.Series(spike.sum(axis=1), index=eng.slots(t0, t1)).groupby(eng.service_day(eng.slots(t0, t1))).sum()
    # hidden closures: runs of zeros where lambda > 20
    Z = (obs.values == 0) & (base.values > 20)
    runs = []
    for j, k in enumerate(eng.keys):
        r = mx = 0
        for i in range(len(obs)):
            r = r + 1 if Z[i, j] else 0; mx = max(mx, r)
        if mx >= 3:
            runs.append({'station': k, 'zero_slots_in_a_row': mx})
    daily, _, _ = _daily(eng)
    En = eng.energy.astype(float).copy(); En.index = pd.to_datetime(En.index).normalize()
    energy = {}
    if day in En.index:
        for Ln in En.columns:
            ks = [k for k in eng.keys if Ln in eng.stations.at[k, 'lines']]
            b0, b1, s = eng.energy_model[Ln]
            energy[Ln] = (En.at[day, Ln] - (b0 + b1 * daily.loc[day, ks].sum())) / s
    # robust findings first: event crowds (several stations/slots), weather episodes, unexplained multi-slot excesses
    findings, spikes = [], []
    for ec in events:
        if ec['z'] >= 3 and ec['extra'] > 0:
            findings.append({'type': 'event crowd', 'when': ec['window'], 'where': ec['stations'], 'observed': ec['observed'],
                             'normal': ec['normal'], 'extra': ec['extra'],
                             'most_likely_cause': f"{ec['event']} at {ec['venue']} ({int(ec['attendance'])} attendees), {ec['phase']} wave",
                             'evidence': f"extra = {ec['extra_share_of_attendance']:.0%} of attendance; strength {ec['z']:.1f} (above 3 is clear)",
                             '_key': 100 + ec['z'], '_ev': f"{ec['event']} | {ec['venue']}"})
    for ep in episodes:
        sdw = ep.get('usual_week_to_week_sd_of_obs_over_model') or 0.03
        if ep['excess_vs_dry'] > max(0.05, 2 * sdw):
            findings.append({'type': 'weather', 'when': ep['window'], 'where': 'whole network', 'observed': ep['observed'],
                             'normal': ep['expected_dry'], 'extra': ep['observed'] - ep['expected_dry'],
                             'most_likely_cause': f"rain up to {ep['max_rain_mm_h']} mm/h with a {ep['temp_drop']:.1f} degC temperature drop, wind up to {ep['wind_max_kmh']:.0f} km/h",
                             'evidence': f"{ep['excess_vs_dry']:+.0%} vs a dry day; the rain rule explains {ep['share_of_excess_explained_by_rain'] or 0:.0%}; "
                                         f"observed/model {ep['obs_over_model']:.3f} vs usual week-to-week spread {sdw:.3f}",
                             '_key': 50})
    for _, r in best.iterrows():
        t_end = r.window_end
        sl = eng.slots(t_end - pd.Timedelta('45min'), t_end)
        o = eng.flows.loc[sl, r.station].values; x = erb.loc[sl, r.station].values
        exc = o - x
        if exc.sum() <= 0:
            continue
        if exc.max() / exc.sum() > 0.6:
            spikes.append(f"{r.station} {int(o[exc.argmax()])} at {sl[exc.argmax()]:%H:%M} (expected about {x[exc.argmax()]:.0f})")
        elif r.z_vs_model > 5 and (o > 3 * np.maximum(x, 1)).sum() >= 2:
            findings.append({'type': 'unexplained', 'when': r.window, 'where': [r.station], 'observed': r.observed, 'normal': r.normal,
                             'extra': r.observed - r.normal, 'most_likely_cause': 'no event, closure or weather explains it',
                             'evidence': f'strength {r.z_vs_model:.1f} after events and weather', '_key': 10 + r.z_vs_model})
    # one finding per cause: the arrival and departure waves of the same event are merged
    merged, by_ev = [], {}
    for f in findings:
        ev = f.pop('_ev', None)
        if ev is None:
            merged.append(f); continue
        if ev not in by_ev:
            by_ev[ev] = f; merged.append(f); f['_phases'] = [f['when']]
            continue
        g = by_ev[ev]
        g['_phases'].append(f['when'])
        g['where'] = list(dict.fromkeys(list(g['where']) + list(f['where'])))
        g['observed'] += f['observed']; g['normal'] += f['normal']; g['extra'] += f['extra']
        g['evidence'] += '; ' + f['evidence']
        g['most_likely_cause'] = g['most_likely_cause'].rsplit(',', 1)[0] + ', arrival and departure waves'
        g['_key'] = max(g['_key'], f['_key'])
    for f in merged:
        if '_phases' in f:
            f['when'] = ' and '.join(f.pop('_phases'))
    findings = merged
    # unusual whole days at a station, against the same weekday in other weeks (closure days excluded)
    daily, _, _ = _daily(eng)
    if day in daily.index:
        closed_days = {}
        for c in eng._clist:
            if c['kind'] == 'station' and c['stations'] and c['start'] is not None:
                closed_days.setdefault(c['stations'][0], set()).add(eng.service_day(pd.DatetimeIndex([c['start']]))[0])
        same = daily[(daily.index.dayofweek == day.dayofweek) & (daily.index != day)]
        in_split = set().union(*[set(eng.venue_shares(e['venue'])[1]) for e in eng._data_events(S, E)]) if eng._data_events(S, E) else set()
        closed_today = {k for k, ds in closed_days.items() if day in ds}
        for k in eng.keys:
            if k in in_split or k in closed_today:
                continue
            ref = same[k][[d not in closed_days.get(k, set()) for d in same.index]]
            if len(ref) < 5 or ref.std() <= 0:
                continue
            z = (daily.at[day, k] - ref.mean()) / ref.std()
            if abs(z) >= 3:
                findings.append({'type': 'station-day', 'when': 'whole day', 'where': [k], 'observed': daily.at[day, k],
                                 'normal': ref.mean(), 'extra': daily.at[day, k] - ref.mean(),
                                 'most_likely_cause': 'no event, closure or rain explains it (possibly an unlisted event or a counting problem)',
                                 'evidence': f'{z:+.1f} standard deviations from the same weekday in {len(ref)} other weeks '
                                             f'(range {ref.min():.0f}-{ref.max():.0f})', '_key': 30 + abs(z)})
    findings.sort(key=lambda f: -f.pop('_key'))
    out = {'date': day, 'weekday': day.day_name(),
           'ranked_findings': findings,
           'isolated_spikes_not_anomalies': {'examples': spikes[:6],
                                             'why': f'single 15-min readings far above expected happen about {per_day.mean():.0f} times a day in this data, rain or not; '
                                                    f'this day had {per_day.get(day)}'},
           'closures_that_day': [f"{c['description']} ({c['start']:%d %b %H:%M}-{c['end']:%d %b %H:%M})" for c in eng._clist
                                 if eng.service_day(pd.DatetimeIndex([c['start']]))[0] == day],
           'closures_ending_that_morning': [c['description'] for c in eng._clist if c['start'] < S <= c['end'] + pd.Timedelta('1h')],
           'largest_1h_excesses': best[['station', 'window', 'observed', 'normal', 'model', 'z_vs_normal', 'z_vs_model']].round(1).to_dict(orient='records'),
           'event_checks': events, 'extra_stations_near_venues': extra_near, 'weather': weather,
           'network_obs_over_model_by_hour': net.groupby(net.index.hour).mean().round(2).to_dict(),
           'isolated_spikes': {'readings_over_8x_expected_this_day': per_day.get(day), 'average_per_day': per_day.mean(),
                               'note': 'isolated spikes at quiet times (midday, late night) are normal in this data, rain or not'},
           'hidden_closures_zero_runs': runs, 'energy_residual_z_by_line': energy}
    return _py(out)


# ---------------------------------------------------------------------------------- 8 dependencies
@lru_cache(maxsize=2)
def _residual_z(eng, with_events: bool):
    t0, t1 = eng.flows.index.min(), eng.flows.index.max()
    lam = eng.expected(t0, t1, events='data' if with_events else [])
    er = _reading(eng, lam)
    obs = eng.flows.reindex(lam.index)
    hour = lam.index.floor('h')
    O, X, V = obs.groupby(hour).sum(), er.groupby(hour).sum(), (er ** 2).groupby(hour).sum()
    Z = ((O - X) / np.sqrt(V + 1)).clip(-10, 10)
    Z = Z[X.sum(axis=1) > 0]
    return Z.sub(Z.mean(axis=1), axis=0)


def station_dependencies(eng: UBahnEngine, top: int = 8):
    """Non-adjacent station pairs whose deviations from normal move together, whether the link survives
    outside event hours and after modelling events, and the venue that drives it."""
    Zb, Zf = _residual_z(eng, False), _residual_z(eng, True)
    C = np.corrcoef(Zb.values.T)
    keys = eng.keys; G = eng.graph
    iu = np.triu_indices(len(keys), 1)
    vals = C[iu]
    order = np.argsort(-vals)
    mask = pd.Series(False, index=Zb.index)
    for r in eng.events.itertuples():
        mask[(mask.index >= (r.start - pd.Timedelta('2h')).floor('h')) & (mask.index <= r.end + pd.Timedelta('2h'))] = True
    out = []
    for o in order:
        a, b = keys[iu[0][o]], keys[iu[1][o]]
        if G.has_edge(a, b):
            continue
        drivers = {}
        for vk, g in eng.events.dropna(subset=['venue_key']).groupby('venue_key'):
            s = 0.0
            for (s0, e0), _ in g.groupby(['start', 'end']):
                w = (Zb.index >= e0.floor('h')) & (Zb.index <= e0 + pd.Timedelta('2h'))
                s += float((Zb.loc[w, a] * Zb.loc[w, b]).sum())
            drivers[vk] = s
        shared = [vk for vk in drivers if a in eng.venue_shares(vk)[1] and b in eng.venue_shares(vk)[1]]
        best_v = max(shared, key=lambda v: drivers[v]) if shared else max(drivers, key=drivers.get)
        out.append({'pair': [a, b], 'lines': [eng.stations.at[a, 'lines'], eng.stations.at[b, 'lines']],
                    'stops_apart': nx.shortest_path_length(G, a, b),
                    'km_apart': float(_hav(eng.stations.at[a, 'latitude'], eng.stations.at[a, 'longitude'], eng.stations.at[b, 'latitude'], eng.stations.at[b, 'longitude'])),
                    'correlation': vals[o], 'outside_event_hours': np.corrcoef(Zb[a][~mask], Zb[b][~mask])[0, 1],
                    'after_event_model': np.corrcoef(Zf[a], Zf[b])[0, 1], 'common_cause_venue': best_v,
                    'both_s_bahn': _sbahn(eng, a) and _sbahn(eng, b)})
        if len(out) >= top:
            break
    return _py({'method': 'hourly deviation from normal (time of day, weather, closures) per station, network-wide swings removed; '
                          'correlation over all hours for all 14,028 pairs',
                'typical_pair_correlation': float(np.median(vals)), 'top_0_1_percent_threshold': float(np.quantile(vals, 0.999)),
                'pairs': out,
                'mechanism': 'shared event catchment: an event sends its crowd to every station within walking distance, '
                             'whatever the line; the stations depend on the venue, not on each other'})


# ---------------------------------------------------------------------------------- 9 disruptions
def disruption_response(eng: UBahnEngine):
    """What passengers actually do during disruptions: pooled response of adjacent / walkable stations to
    station closures and of section / alternative stations to line suspensions, plus the rebound."""
    closures, susp = [], []
    agg = {'adjacent': [0, 0, 0], 'walkable': [0, 0, 0]}
    lost_total = 0
    best_example = None
    for c in eng._clist:
        S, E = c['start'].ceil('15min'), c['end'] - pd.Timedelta('1min')
        if c['kind'] == 'station':
            k = c['stations'][0]
            nb = list(eng.graph.neighbors(k))
            walk = [s for s, d in eng.stations_near(eng.stations.at[k, 'latitude'], eng.stations.at[k, 'longitude'], 1.2) if s != k and s not in nb]
            lost = _reading(eng, eng.expected(S, E, [k], closures=[])).values.sum(); lost_total += lost
            rec = {'station': k, 'start': c['start'], 'hours': (c['end'] - c['start']).total_seconds() / 3600, 'lost': round(lost)}
            for name, sts in (('adjacent', nb), ('walkable', walk)):
                if sts:
                    lam = eng.expected(S, E, sts); er = _reading(eng, lam); ob = eng.flows.reindex(lam.index)[sts]
                    ex, sd, ee = ob.values.sum() - er.values.sum(), (er.values ** 2).sum(), er.values.sum()
                    agg[name][0] += ex; agg[name][1] += sd; agg[name][2] += ee
                    rec[f'{name}_change'] = ex / ee
            closures.append(rec)
            if best_example is None or (lost > best_example[1] and 16 <= c['start'].hour <= 18):
                best_example = (c, lost)
        elif c['kind'] == 'line' and c['stations']:
            L, sec = c['line'], c['stations']
            ro = eng.reroute_options(L, sec[0], sec[-1])
            alts = sorted({k for s in ro['interior_without_service'] for k, _ in ro['per_station_alternatives'][s]['walkable_alternatives_km']} - set(sec))
            rec = {'description': c['description'], 'start': c['start']}
            for name, sts in (('interior', ro['interior_without_service']), ('section_ends', [sec[0], sec[-1]]), ('alternatives', alts)):
                if sts:
                    lam = eng.expected(S, E, sts); er = _reading(eng, lam); ob = eng.flows.reindex(lam.index)[sts]
                    ex, sd, ee = ob.values.sum() - er.values.sum(), (er.values ** 2).sum(), er.values.sum()
                    a = agg.setdefault('susp_' + name, [0, 0, 0]); a[0] += ex; a[1] += sd; a[2] += ee
                    rec[name + '_change'] = ex / ee
            susp.append(rec)
    pooled = {k: {'extra_passengers': v[0], 'expected': v[2], 'change': v[0] / max(v[2], 1), 'z': v[0] / np.sqrt(max(v[1], 1))}
              for k, v in agg.items()}
    ex = None
    if best_example:
        c, lost = best_example; k = c['stations'][0]
        t = c['end'].ceil('15min'); after = eng.slots(t, t + pd.Timedelta('30min'))
        nb = list(eng.graph.neighbors(k))
        w = _window(eng, nb, c['start'].ceil('15min'), c['end'] - pd.Timedelta('1min')).sum()
        normal = _reading(eng, eng.expected(after[0], after[-1], [k], closures=[]))[k]
        ex = {'station': k, 'closed': f"{c['start']:%a %d %b %H:%M}-{c['end']:%H:%M}", 'passengers_lost': round(lost),
              'neighbours': nb, 'neighbours_observed': w.observed, 'neighbours_expected': w.model,
              'after_reopening': [{'slot': x, 'observed': eng.flows.at[x, k], 'normal': round(normal[x]), 'ceiling': eng.ceiling[k]} for x in after]}
    return _py({'station_closures': closures, 'line_suspensions': susp, 'pooled': pooled,
                'passengers_displaced_by_station_closures': lost_total,
                'rebound_at_reopened_station': {'share_of_lost_volume_back_within_45_min': 0.56, 'per_slot': [0.39, 0.12, 0.05],
                                                'source': 'estimated from the recorded reopenings and built into the engine; it describes '
                                                          'how the (simulated) data behave, not observed passenger preferences'},
                'example': ex,
                'interpretation': ['in this simulated data, flows shift in time rather than in space: most of the lost volume '
                                   'reappears at the same station after it reopens',
                                   'about 44 % of displaced trips never reappear in the U-Bahn data (abandoned or other modes)',
                                   'the simulator has no route choice: shortest-route planning predicts spill-over that never happens']})


# ---------------------------------------------------------------------------------- 10 investment
@lru_cache(maxsize=2)
def _vulnerability(eng, extra_edge=None):
    G = eng.graph.copy()
    if extra_edge:
        G.add_edge(*extra_edge)
    n = G.number_of_nodes(); _, wd, _ = _daily(eng)
    tp = tx = 0; per = {}
    for v in G.nodes():
        H = G.copy(); H.remove_node(v)
        comps = sorted(nx.connected_components(H), key=len, reverse=True)
        cut = set().union(*comps[1:]) if len(comps) > 1 else set()
        pl = (n - 1) * (n - 2) // 2 - sum(len(c) * (len(c) - 1) // 2 for c in comps)
        px = wd[list(cut)].sum() if cut else 0.0
        tp += pl; tx += px; per[v] = (len(cut), px)
    return tp, tx, per


def best_new_link(eng: UBahnEngine, max_km: float = 1.3, top: int = 5):
    """Test every short link (<= max_km) between stations on different lines and rank them by how much they
    reduce passengers cut off across all possible single-station closures. Also returns disruption history
    and energy context for the winner's lines. Takes ~20 s."""
    G = eng.graph
    P0, X0, per0 = _vulnerability(eng)
    res = []
    for a, b in itertools.combinations(eng.keys, 2):
        if G.has_edge(a, b):
            continue
        if set(eng.stations.at[a, 'lines']) & set(eng.stations.at[b, 'lines']):
            continue
        d = float(_hav(eng.stations.at[a, 'latitude'], eng.stations.at[a, 'longitude'], eng.stations.at[b, 'latitude'], eng.stations.at[b, 'longitude']))
        if d > max_km:
            continue
        P, X, _ = _vulnerability(eng, (a, b))
        res.append({'link': [a, b], 'km': d, 'exposure_reduction': (X0 - X) / X0, 'pairs_reduction': (P0 - P) / P0})
    res.sort(key=lambda r: -r['exposure_reduction'])
    best = tuple(res[0]['link'])
    _, _, per1 = _vulnerability(eng, best)
    changed = sorted(((v, per0[v], per1[v]) for v in per0 if per0[v][0] != per1[v][0]), key=lambda x: -x[1][1])[:6]
    per_line = {}
    for c in eng._clist:
        lines = [c['line']] if c['kind'] == 'line' else eng.stations.at[c['stations'][0], 'lines']
        for L in lines:
            p = per_line.setdefault(L, [0, 0.0]); p[0] += 1; p[1] += (c['end'] - c['start']).total_seconds() / 3600
    return _py({'method': 'score = reduction of weekday passengers cut off, summed over every possible single-station closure',
                'scope': f'best among the {len(res)} candidate links tested: pairs of stations on different lines within {max_km} km '
                         '(straight line) not already connected',
                'not_evaluated': 'construction cost, feasibility and induced demand',
                'passenger_counts': 'passengers_before/after count the stations cut off only; fragmentation_ranking adds the closed '
                                    'station\'s own passengers (for Alexanderplatz, about 7,800 on a weekday)',
                'candidates_tested': len(res), 'ranking': res[:top],
                'effect_of_best_link': [{'if_closed': v, 'stations_cut_off_before': b[0], 'passengers_before': b[1],
                                         'stations_cut_off_after': a[0], 'passengers_after': a[1]} for v, b, a in changed],
                'disruptions_by_line_count_hours': {k: [v[0], round(v[1], 1)] for k, v in sorted(per_line.items())},
                'route_examples_after': {f'{o} -> {d}': [nx.shortest_path_length(G, o, d), nx.shortest_path_length(nx.Graph(list(G.edges()) + [best]), o, d)]
                                         for o, d in [('Lichtenberg Bhf', 'Kottbusser Tor'), ('Hönow', 'Kurfürstendamm')]}})


# ---------------------------------------------------------------------------------- 11 diversion
def diversion_scenario(eng: UBahnEngine, venue: str, start: str, end: str, attendance: float,
                       alternative_stations: list | None = None, destinations: list | None = None,
                       shares: tuple = (0.0, 0.25, 0.4), weather: dict | None = None, max_walk_km: float = 2.6):
    """Send part of an event crowd to alternative stations (e.g. by shuttle bus or a walk) instead of the
    venue's own stations, and compare peak demand / ceiling for the exit wave. Also gives the default route
    and an alternative route that avoids the default corridor."""
    s0, e0 = _to_dt(start), _to_dt(end)
    key, shares_v = eng.venue_shares(venue)
    home = list(shares_v)
    main_lines = set(sum([eng.stations.at[s, 'lines'] for s in home if shares_v[s] >= 0.1], []))
    auto = not alternative_stations
    if auto:
        # another corridor: nearest stations within walking/shuttle distance that share no line with the venue's main stations
        _, vlat, vlon = eng.geocode(venue)
        cands = [(s, d) for s, d in eng.stations_near(vlat, vlon, max_walk_km)
                 if s not in home and not (set(eng.stations.at[s, 'lines']) & main_lines)]
        if not cands:
            return {'error': f'no station on another line within {max_walk_km} km of the venue; pass alternative_stations'}
        by_line = {}
        for s_, d in cands:
            for L in eng.stations.at[s_, 'lines']:
                by_line.setdefault(L, []).append((d, s_))
        best_line = min(by_line, key=lambda L: sorted(by_line[L])[0][0])
        alternative_stations = [s_ for _, s_ in sorted(by_line[best_line])[:2]]
    alts = [eng.resolve_station(s) for s in alternative_stations]
    same_line = [f"{s_} is on {', '.join(sorted(set(eng.stations.at[s_, 'lines']) & main_lines))} like the venue's stations: trains reach it "
                 f"already loaded, so it spreads platform crowding but adds no line capacity" for s_ in alts
                 if set(eng.stations.at[s_, 'lines']) & main_lines]
    lat = float(np.mean([eng.stations.at[s, 'latitude'] for s in alts])); lon = float(np.mean([eng.stations.at[s, 'longitude'] for s in alts]))
    watch = list(dict.fromkeys(home + alts))
    idx0, idx1 = e0 - MIN15, e0 + pd.Timedelta('2h')
    if weather is None and not np.isfinite(eng._daily_temp(eng.slots(idx0, idx1))).all():
        weather = {'tmean': 18, 'prcp': 0}
    cap = eng.ceiling[watch].values

    def ratios(sh):
        evs = [{'venue': venue, 'start': s0, 'end': e0, 'attendance': attendance * (1 - sh)}]
        if sh:
            evs.append({'venue': (lat, lon), 'start': s0, 'end': e0, 'attendance': attendance * sh})
        lam = eng.expected(idx0, idx1, watch, weather=weather, events=[], extra_events=evs, closures=[])
        return dict(zip(watch, (lam.max() / cap).round(2)))
    table = {f'{int(sh * 100)}%_diverted': ratios(sh) for sh in shares}
    grid = {round(x, 2): ratios(x) for x in np.arange(0, 0.61, 0.05)}
    ok = [x for x, r_ in grid.items() if all(r_[a_] <= 1.0 for a_ in alts)]
    cl = max(ok) if ok else 0.0
    capacity_limited = {'share': cl, 'ratios': grid[cl],
                        'note': 'largest diverted share that keeps every alternative station at or below its ceiling at the peak; '
                                'if the venue stations stay above their ceiling at that share, diversion alone is not enough'}
    lam0 = eng.expected(idx0, idx1, watch, weather=weather, events=[], closures=[])
    table['no_event'] = dict(zip(watch, (lam0.max() / cap).round(2)))
    dests = destinations or ['Alexanderplatz Bhf', 'Friedrichstr. Bhf', 'Potsdamer Platz Bhf']
    routes = []
    for d in dests:
        dflt = route(eng, home[0], d, _with_rides=True)
        first_leg = [r for r in dflt.get('_rides', []) if r[2] == dflt['_rides'][0][2]] if dflt.get('_rides') else []
        for a in alts[:2]:
            alt = route(eng, a, d, avoid_line_segments=first_leg, _with_rides=True)
            routes.append({'destination': d, 'default': {k: v for k, v in dflt.items() if k != '_rides'},
                           'alternative': {k: v for k, v in alt.items() if k != '_rides'}})
    walk = {s: round(float(_hav(*eng.geocode(venue)[1:], eng.stations.at[s, 'latitude'], eng.stations.at[s, 'longitude'])), 2) for s in watch}
    return _py({'venue': key, 'venue_stations': home, 'venue_lines': sorted(main_lines), 'alternative_stations': alts,
                'alternatives_chosen_by_tool': auto, 'same_line_warning': same_line, 'walk_km_from_venue': walk,
                'status': 'option to investigate, not an operating instruction',
                'capacity_limited_share': capacity_limited,
                'access': {a_: {'straight_line_km': walk[a_], 'walk_minutes_at_4_5_kmh': round(walk[a_] / 4.5 * 60),
                                'needs_shuttle_or_guided_route': walk[a_] > 1.5} for a_ in alts},
                'peak_demand_over_ceiling_exit_wave': table, 'routes': routes,
                'note': 'above 1 = demand beyond the station ceiling; passengers in this data never reroute on their own, '
                        'so a diversion needs shuttles, signage, staff and app messages',
                's_bahn_at': [s for s in watch if _sbahn(eng, s)]})


# ================================================================================== generic building blocks
# Small, composable tools for plain data questions (how many, average, busiest, compare, list). They make
# no analytical choice beyond the one stated in their arguments, so the model can combine them freely.

DAY_TYPES = ('all', 'weekday', 'saturday', 'sunday', 'weekend')


def _scope(eng, stations=None, line=None):
    """Resolve the stations a question is about: a list of names, a line, or the whole network."""
    if stations:
        keys = []
        for s in stations:
            keys += eng.resolve_station(s, multi=True)
        return list(dict.fromkeys(keys)), f'{len(keys)} station(s)'
    if line:
        L = line.upper().replace(' ', '')
        if L not in eng.line_seq:
            raise ValueError(f'{line} is not in this network (lines: {sorted(eng.line_seq)})')
        return list(eng.line_seq[L]), f'all {len(eng.line_seq[L])} stations of {L}'
    return list(eng.keys), 'whole network (168 stations)'


def _minutes_from_5(hhmm: str) -> int:
    h, m = (int(x) for x in hhmm.split(':'))
    return (h * 60 + m - 300) % 1440


def _day_mask(eng, index, days='all', date_from=None, date_to=None, dates=None):
    sd = eng.service_day(index)
    dt = eng.daytype(sd)
    m = np.ones(len(index), bool)
    if dates:
        m &= np.isin(sd, pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in dates]))
    if date_from:
        m &= sd >= pd.Timestamp(date_from).normalize()
    if date_to:
        m &= sd <= pd.Timestamp(date_to).normalize()
    if days not in DAY_TYPES:
        raise ValueError(f'days must be one of {DAY_TYPES}')
    if days == 'weekday':
        m &= dt == 0
    elif days == 'saturday':
        m &= (dt == 1)
    elif days == 'sunday':
        m &= (dt == 2)
    elif days == 'weekend':
        m &= dt > 0
    return m, sd


def _time_mask(index, time_from=None, time_to=None):
    """Time-of-day window [time_from, time_to) in service-day order (05:00 -> 00:45)."""
    mins = np.array([_minutes_from_5(t) for t in index.strftime('%H:%M')])
    a = _minutes_from_5(time_from) if time_from else 0
    b = _minutes_from_5(time_to) if time_to else 1440
    if time_to and b == 0:
        b = 1440
    return (mins >= a) & (mins < b) if a < b else (mins >= a) | (mins < b)


def flow_stats(eng: UBahnEngine, stations: list | None = None, line: str | None = None,
               start: str | None = None, end: str | None = None,
               time_from: str | None = None, time_to: str | None = None,
               date_from: str | None = None, date_to: str | None = None, days: str = 'all', dates: list | None = None,
               by: str = 'total', stat: str | None = None):
    """Recorded passenger flow for stations / a line / the network, in one of two modes.

    Absolute window: start and end (end excluded), e.g. '2026-09-01 00:00' to '2026-09-01 12:00' gives the
        total of the slots 00:00 ... 11:45 of that calendar date (default stat = sum).
    Typical day: a daily time window (time_from, time_to; service-day order, so '22:00'-'01:00' works) over
        a set of service days (date_from, date_to, days = all | weekday | saturday | sunday | weekend, or an
        explicit list of dates). Default stat = mean over the selected days of the window total.
    by = total | slot | hour | day breaks the result down (per 15-min slot, per hour, per service day).
    stat = sum | mean | median | max across days (typical mode) or across slots (absolute mode, by=total)."""
    keys, scope = _scope(eng, stations, line)
    F = eng.flows[keys]
    notes = ['slots are labelled by their start time; 01:00-04:45 has no service']
    if start or end:
        s0 = pd.Timestamp(start) if start else F.index.min()
        e0 = pd.Timestamp(end) if end else F.index.max() + MIN15
        sub = F[(F.index >= s0) & (F.index < e0)]
        if sub.empty:
            return {'error': f'no data between {s0} and {e0}', 'data_range': f'{F.index.min()} -> {F.index.max()}'}
        per_slot = sub.sum(axis=1)
        out = {'mode': 'absolute window', 'scope': scope, 'from': s0, 'to_excluded': e0, 'slots': len(sub),
               'total': per_slot.sum(), 'mean_per_slot': per_slot.mean(), 'max_slot': [per_slot.idxmax(), per_slot.max()],
               'slots_at_ceiling': int((sub >= eng.ceiling[keys]).values.sum())}
        if len(keys) <= 12:
            out['per_station_total'] = sub.sum().to_dict()
        if by == 'slot':
            out['by_slot'] = {t.strftime('%Y-%m-%d %H:%M'): v for t, v in per_slot.items()}
        elif by == 'hour':
            h = per_slot.groupby(per_slot.index.floor('h')).sum()
            out['by_hour'] = {t.strftime('%Y-%m-%d %H:00'): v for t, v in h.items()}
        elif by == 'day':
            d = per_slot.groupby(eng.service_day(per_slot.index)).sum()
            out['by_service_day'] = {t.strftime('%Y-%m-%d'): v for t, v in d.items()}
        if ((sub.index.hour < 1) & (sub.index.minute <= 45)).any():
            notes.append('slots 00:00-00:45 belong to the previous service day (late-night service)')
        out['notes'] = notes
        return _py(out)
    # typical-day mode
    m, sd = _day_mask(eng, F.index, days, date_from, date_to, dates)
    m &= _time_mask(F.index, time_from, time_to)
    sub = F[m]
    if sub.empty:
        return {'error': 'no slot matches these filters'}
    per_slot = sub.sum(axis=1)
    sdays = eng.service_day(per_slot.index)
    stat = stat or 'mean'
    agg = {'sum': 'sum', 'mean': 'mean', 'median': 'median', 'max': 'max'}[stat]
    daily = per_slot.groupby(sdays).sum()
    out = {'mode': 'typical day', 'scope': scope, 'days': days, 'service_days': int(daily.size),
           'date_range': [daily.index.min(), daily.index.max()],
           'window': f"{time_from or '05:00'}-{time_to or 'end of service'}", 'stat_across_days': stat,
           'window_total': getattr(daily, agg)(), 'window_total_min_max': [daily.min(), daily.max()]}
    if by == 'slot':
        prof = per_slot.groupby(per_slot.index.strftime('%H:%M')).agg(agg)
        out['by_slot'] = {t: prof[t] for t in TOD_ORDER if t in prof.index}
    elif by == 'hour':
        hourly = per_slot.groupby([sdays, per_slot.index.hour]).sum().groupby(level=1).agg(agg)
        out['by_hour'] = {f'{h:02d}:00': hourly[h] for h in list(range(5, 24)) + [0] if h in hourly.index}
    elif by == 'day':
        out['by_service_day'] = {t.strftime('%Y-%m-%d'): v for t, v in daily.items()}
    if len(keys) <= 12:
        per_st = sub.groupby(eng.service_day(sub.index)).sum()
        out['per_station_window_total'] = getattr(per_st, agg)().to_dict()
    notes.append('typical-day windows follow the service day (05:00 -> 00:45 next morning)')
    out['notes'] = notes
    return _py(out)


def rank_stations(eng: UBahnEngine, metric: str = 'total', start: str | None = None, end: str | None = None,
                  time_from: str | None = None, time_to: str | None = None, date_from: str | None = None,
                  date_to: str | None = None, days: str = 'all', line: str | None = None, top: int = 10,
                  ascending: bool = False):
    """Rank stations over a period. metric = total (passengers), mean_per_day, peak_slot (highest 15-min
    reading), slots_at_ceiling, vs_expected (observed / model expectation incl. weather, events, closures).
    Period: absolute (start, end excluded) or typical-day filters as in ubahn_flow_stats."""
    keys, scope = _scope(eng, None, line)
    F = eng.flows[keys]
    if start or end:
        s0 = pd.Timestamp(start) if start else F.index.min(); e0 = pd.Timestamp(end) if end else F.index.max() + MIN15
        m = (F.index >= s0) & (F.index < e0)
    else:
        m, _ = _day_mask(eng, F.index, days, date_from, date_to)
        m &= _time_mask(F.index, time_from, time_to)
    sub = F[m]
    if sub.empty:
        return {'error': 'no slot matches these filters'}
    ndays = eng.service_day(sub.index).nunique()
    if metric == 'total':
        val = sub.sum()
    elif metric == 'mean_per_day':
        val = sub.sum() / ndays
    elif metric == 'peak_slot':
        val = sub.max()
    elif metric == 'slots_at_ceiling':
        val = (sub >= eng.ceiling[keys]).sum()
    elif metric == 'vs_expected':
        idx = sub.index
        lam = eng.expected(idx.min(), idx.max(), keys).reindex(idx)
        er = eng.expected_reading(lam.values, eng.ceiling[keys].values[None, :])
        val = pd.Series(sub.values.sum(axis=0) / np.maximum(er.sum(axis=0), 1), index=keys)
    else:
        raise ValueError('metric must be total, mean_per_day, peak_slot, slots_at_ceiling or vs_expected')
    order = val.sort_values(ascending=ascending).head(top)
    rows = []
    for k, v in order.items():
        r = {'station': k, 'lines': eng.stations.at[k, 'lines'], metric: v}
        if metric == 'peak_slot':
            r['at'] = sub[k].idxmax()
        if metric in ('peak_slot', 'slots_at_ceiling'):
            r['ceiling'] = eng.ceiling[k]
        rows.append(r)
    return _py({'metric': metric, 'scope': scope, 'slots': len(sub), 'service_days': ndays,
                'period': [sub.index.min(), sub.index.max()], 'ranking': rows,
                'network_median': val.median()})


def compare_to_normal(eng: UBahnEngine, start: str, end: str, stations: list | None = None, line: str | None = None):
    """Recorded flow over [start, end) compared with two baselines: the model's expectation (with and without
    events, weather and closures included) and the average of the same weekday and hours in the other weeks."""
    keys, scope = _scope(eng, stations, line)
    s0, e0 = pd.Timestamp(start), pd.Timestamp(end)
    idx = eng.flows.index[(eng.flows.index >= s0) & (eng.flows.index < e0)]
    if len(idx) == 0:
        return {'error': f'no data between {s0} and {e0}'}
    obs = eng.flows.loc[idx, keys]
    lam = eng.expected(idx.min(), idx.max(), keys).reindex(idx)
    lam0 = eng.expected(idx.min(), idx.max(), keys, events=[]).reindex(idx)
    cap = eng.ceiling[keys].values[None, :]
    exp_m = eng.expected_reading(lam.values, cap).sum(axis=0)
    exp_0 = eng.expected_reading(lam0.values, cap).sum(axis=0)
    weeks = []
    for w in range(-16, 17):
        if w == 0:
            continue
        a, b = s0 + pd.Timedelta(weeks=w), e0 + pd.Timedelta(weeks=w)
        if a < eng.flows.index.min() or b > eng.flows.index.max() + MIN15:
            continue
        o = eng.flows[(eng.flows.index >= a) & (eng.flows.index < b)][keys].sum()
        weeks.append(o)
    same = pd.concat(weeks, axis=1) if weeks else None
    tot_obs = obs.values.sum()
    out = {'scope': scope, 'window': [s0, e0], 'weekday': s0.day_name(), 'observed': tot_obs,
           'model_expected': exp_m.sum(), 'model_expected_without_events': exp_0.sum(),
           'observed_over_model': tot_obs / max(exp_m.sum(), 1)}
    if same is not None:
        tot_same = same.sum(axis=0)
        out.update({'same_weekday_other_weeks': {'weeks': int(same.shape[1]), 'mean': tot_same.mean(),
                                                 'min': tot_same.min(), 'max': tot_same.max()},
                    'observed_over_same_weekday_mean': tot_obs / max(tot_same.mean(), 1),
                    'share_of_other_weeks_with_higher_flow': float((tot_same > tot_obs).mean())})
    if len(keys) <= 12:
        out['per_station'] = [{'station': k, 'observed': obs[k].sum(), 'model_expected': round(float(e)),
                               'same_weekday_mean': round(float(same.loc[k].mean())) if same is not None else None,
                               'slots_at_ceiling': int((obs[k] >= eng.ceiling[k]).sum())}
                              for k, e in zip(keys, exp_m)]
    out['events_in_window'] = [f"{e['name']} ({e['start']:%H:%M}-{e['end']:%H:%M}, {int(e['attendance'])}) @ {eng.geocode(e['venue'])[0]}"
                               for e in eng._data_events(idx.min(), idx.max())][:15]
    out['closures_in_window'] = [c['description'] for c in eng._clist if c['start'] < e0 and c['end'] > s0]
    out['note'] = 'single stations vary a lot from week to week; compare with the min-max of the other weeks'
    return _py(out)


def list_events(eng: UBahnEngine, date_from: str | None = None, date_to: str | None = None, venue: str | None = None,
                near_station: str | None = None, radius_km: float = 1.5, min_attendance: float = 0, limit: int = 30):
    """Events from the events file, grouped by venue and start time (listings of the same show added up),
    with the stations their crowds use."""
    ev = eng.events.copy()
    sd = eng.service_day(ev.start)
    m = np.ones(len(ev), bool)
    if date_from:
        m &= sd >= pd.Timestamp(date_from).normalize()
    if date_to:
        m &= sd <= pd.Timestamp(date_to).normalize()
    if venue:
        q = venue.lower()
        key = eng.geocode(venue)[0]
        m &= (ev.event_name.str.lower().str.contains(q, regex=False) | ev.venue_name.fillna('').str.lower().str.contains(q, regex=False)
              | ev.address.str.lower().str.contains(q, regex=False) | (ev.venue_key == key if key else False)).values
    ev = ev[m]
    if near_station:
        k = eng.resolve_station(near_station)
        lat, lon = eng.stations.at[k, 'latitude'], eng.stations.at[k, 'longitude']
        keep = []
        for r in ev.itertuples():
            vk, vlat, vlon = eng.geocode(r.address, r.venue_name if isinstance(r.venue_name, str) else '')
            keep.append(bool(vk) and _hav(lat, lon, vlat, vlon) <= radius_km)
        ev = ev[np.array(keep, bool)] if len(ev) else ev
    rows = []
    for (vk, s0), g in ev.groupby(['venue_key', 'start'], dropna=False):
        att = g.estimated_attendance.sum()
        if att < min_attendance:
            continue
        _, shares = eng.venue_shares(g.address.iloc[0] + ' ' + str(g.venue_name.iloc[0] if isinstance(g.venue_name.iloc[0], str) else ''))
        rows.append({'start': s0, 'end': g.end.max(), 'weekday': s0.day_name(), 'names': sorted(g.event_name.unique())[:3],
                     'venue': vk, 'address': g.address.iloc[0], 'attendance': att, 'listings': len(g),
                     'stations': {k: round(v, 2) for k, v in sorted(shares.items(), key=lambda x: -x[1])[:4]}})
    rows.sort(key=lambda r: r['start'])
    return _py({'events_found': len(rows), 'shown': min(len(rows), limit), 'events': rows[:limit],
                'note': 'end times in the data are estimates (start + 3 h for music, + 2 h otherwise)'})


def list_closures(eng: UBahnEngine, date_from: str | None = None, date_to: str | None = None, line: str | None = None,
                  station: str | None = None, kind: str | None = None):
    """Closures and suspensions from the closures file, parsed: kind, line, stations, start, end, duration, reason."""
    keys = eng.resolve_station(station, multi=True) if station else None
    rows = []
    for c in eng._clist:
        sd = eng.service_day(pd.DatetimeIndex([c['start']]))[0]
        if date_from and sd < pd.Timestamp(date_from).normalize():
            continue
        if date_to and sd > pd.Timestamp(date_to).normalize():
            continue
        if line and c['line'] != line.upper():
            continue
        if kind and c['kind'] != kind:
            continue
        if keys and not set(keys) & set(c['stations']):
            continue
        dur = c['end'] - c['start']
        reason = re.search(r'due to (.+?)\.?$', c['description'])
        rows.append({'start': c['start'], 'end': c['end'], 'weekday': c['start'].day_name(),
                     'duration': f'{int(dur.total_seconds() // 3600)} h {int(dur.total_seconds() % 3600 // 60):02d} min',
                     'kind': c['kind'], 'line': c['line'], 'stations': c['stations'] if len(c['stations']) <= 12 else c['stations'][:3] + ['...'],
                     'reason': reason.group(1) if reason else None, 'description': c['description'], 'issues': c['issues']})
    rows.sort(key=lambda r: r['start'])
    return _py({'closures_found': len(rows), 'closures': rows,
                'note': 'station closures set flows to zero; line suspensions leave no trace in the flow data'})


def station_info(eng: UBahnEngine, station: str, radius_km: float = 1.0):
    """Facts about one station: lines, S-Bahn interchange, neighbours on the network, stations and event
    venues within walking distance, ceiling and typical traffic."""
    k = eng.resolve_station(station)
    st = eng.stations.loc[k]
    daily, wd, we = _daily(eng)
    F = eng.flows[k]
    dt = eng.daytype(eng.service_day(F.index))
    prof = F[dt == 0].groupby(F[dt == 0].index.strftime('%H:%M')).mean().reindex(TOD_ORDER)
    venues = []
    for vk, (vlat, vlon, _) in VENUES_REF().items():
        d = float(_hav(st.latitude, st.longitude, vlat, vlon))
        if d <= max(radius_km, 1.5):
            shares = eng.venue_shares((vlat, vlon))[1] if vk not in getattr(eng, 'venue_learned', {}) else eng.venue_learned[vk]
            venues.append({'venue': vk, 'km': round(d, 2), 'share_of_its_crowd_using_this_station': round(shares.get(k, 0.0), 2)})
    venues.sort(key=lambda v: v['km'])
    return _py({'station': k, 'official_name': st.station_name, 'station_id': st.station_id, 'lines': st.lines,
                's_bahn_interchange': _sbahn(eng, k), 'lat_lon': [st.latitude, st.longitude],
                'adjacent_stations': [{'station': n, 'lines': sorted(set(eng.stations.at[n, 'lines']) & set(st.lines))}
                                      for n in eng.graph.neighbors(k)],
                'within_walking_distance': [(s, d) for s, d in eng.stations_near(st.latitude, st.longitude, radius_km) if s != k],
                'ceiling_15min': eng.ceiling[k],
                'typical_daily_passengers': {'weekday': wd[k], 'weekend_day': we[k]},
                'typical_weekday_peaks': {'morning': [prof.loc['06:00':'10:00'].idxmax(), prof.loc['06:00':'10:00'].max()],
                                          'evening': [prof.loc['15:00':'20:00'].idxmax(), prof.loc['15:00':'20:00'].max()]},
                'event_venues_nearby': venues[:8],
                'closures_in_data': [f"{c['description']} ({c['start']:%d %b %H:%M})" for c in eng._clist if k in c['stations']]})


def VENUES_REF():
    from src.engine import VENUES
    return VENUES


def weather(eng: UBahnEngine, date_from: str, date_to: str | None = None, by: str = 'day'):
    """Weather for a date or period (by day or by hour) and the effect the model attributes to it
    (flow multiplier: 1.0 = 18 degC daily mean and dry)."""
    d0 = pd.Timestamp(date_from).normalize()
    d1 = pd.Timestamp(date_to).normalize() if date_to else d0
    wx = eng.weather.loc[d0 + pd.Timedelta('5h'):d1 + pd.Timedelta('1D 45min')].copy()
    if wx.empty:
        return {'error': 'no weather data for these dates', 'weather_range': f'{eng.weather.index.min():%Y-%m-%d} -> {eng.weather.index.max():%Y-%m-%d}'}
    tday = eng._daily_temp(wx.index)
    wx['multiplier'] = eng.weather_multiplier(tday, wx.prcp.values)
    sd = eng.service_day(wx.index)
    if by == 'hour':
        g = wx.groupby(wx.index.floor('h')).agg(temp=('temp', 'mean'), rain_mm_h=('prcp', 'mean'), wind_kmh=('wspd', 'mean'),
                                                  condition_code=('coco', 'max'), flow_multiplier=('multiplier', 'mean'))
        rows = {t.strftime('%Y-%m-%d %H:00'): r for t, r in g.round(2).iterrows()}
    else:
        g = wx.groupby(sd).agg(temp_mean=('temp', 'mean'), temp_min=('temp', 'min'), temp_max=('temp', 'max'),
                               rain_mm=('prcp', lambda x: x.sum() / 4), max_rain_mm_h=('prcp', 'max'),
                               wind_max_kmh=('wspd', 'max'), condition_code_max=('coco', 'max'),
                               flow_multiplier_mean=('multiplier', 'mean'))
        rows = {t.strftime('%Y-%m-%d (%a)'): r for t, r in g.round(2).iterrows()}
    return _py({'by': by, 'rows': {k: v.to_dict() for k, v in rows.items()},
                'note': 'flow multiplier = model effect of heat (-1.8 % per degC of daily mean above 18) and rain (up to +34 %); '
                        'condition codes follow Meteostat (7-9 rain, 17-18 showers, 25-27 thunderstorm)'})
