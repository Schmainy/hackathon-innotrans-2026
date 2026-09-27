"""Deterministische Antwort aus Tool-Ergebnissen – ohne LLM.

Greift, wenn das LLM das Zeitlimit reisst oder nicht erreichbar ist (Messe-
WLAN). Die Zahlen kommen unverändert aus den Tools; es gibt keine
Interpretation und keine Empfehlungen, nur eine lesbare Zusammenfassung.
Jede Tool-Funktion hat hier einen eigenen Formatierer, der ihre tatsächlichen
Rückgabefelder kennt; unbekannte Tools fallen auf skalare Top-Level-Werte
zurück.
"""

from __future__ import annotations

from typing import Any, Callable

MAX_ITEMS = 5


def _num(value: Any, digits: int = 0) -> str:
    """Tausendertrennung für Zahlen, alles andere unverändert."""
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:,.{digits}f}"
    return str(value)


def _short(name: Any) -> str:
    return str(name).replace(" (Berlin)", "")


def _fmt(value: Any, digits: int = 0) -> str:
    """Englische Zahlschreibweise: 2,849 bzw. 95.0."""
    if value is None or isinstance(value, bool):
        return "–"
    return f"{float(value):,.{digits}f}"


def _signed_pct(value: Any) -> str:
    if value is None:
        return "–"
    return ("+" if value >= 0 else "") + _fmt(value, 1) + "%"


# Ampel zu capacity_status aus flow_tool (Anteil am Stations-Allzeitmaximum).
AMPEL = {
    "CRITICAL": "⚠️ CRITICAL",
    "ELEVATED": "🟡 ELEVATED",
    "NORMAL": "🟢 NORMAL",
}
BOX_WIDTH = 52


def _ampel(capacity_status: str | None) -> str:
    return AMPEL.get(str(capacity_status), str(capacity_status or ""))


def _peak_box(station: Any, peak_time: Any, peak_value: Any, *,
              station_max_ever: Any = None, pct_of_own_max: Any = None,
              capacity_status: str | None = None, date: Any = None, baseline: Any = None, baseline_label: str = "",
              increase_pct: Any = None, confounder: str | None = None,
              extra: list[str] | None = None) -> list[str]:
    """Operator-Block: Peak-Slot zuerst, Einordnung danach.

    Ohne station_max_ever (z. B. get_station_flow) nur der absolute Wert –
    ein relativer Anteil braucht das Stationsmaximum als Referenz.
    """
    head = f"┌─ PEAK LOAD: {_short(station).upper()} "
    rows = [head + "─" * max(4, BOX_WIDTH - len(head))]
    when = f"{peak_time} (15-min slot)"
    if date:
        when += f", {date}"
    rows.append(f"│ Peak time:    {when}")
    load = f"{_fmt(peak_value)}"
    if station_max_ever is not None and pct_of_own_max is not None:
        load += (f"  {_ampel(capacity_status)} ({_fmt(pct_of_own_max, 1)}% of "
                 f"station max {_fmt(station_max_ever)})")
    rows.append(f"│ Passengers:   {load}")
    if baseline is not None:
        rows.append(f"│ Baseline:     {_fmt(baseline)}" + (f" ({baseline_label})" if baseline_label else ""))
        rows.append(f"│ Deviation:    {_signed_pct(increase_pct)} "
                    + ("above" if (increase_pct or 0) >= 0 else "below") + " baseline")
    if confounder:
        rows.append(f"│ Confounders:  {confounder}")
    for line in extra or []:
        rows.append(f"│ {line}")
    rows.append("└" + "─" * (BOX_WIDTH - 1))
    return ["```text", *rows, "```"]


def _station_flow(r: dict) -> list[str]:
    peak = r.get("peak_slot") or {}
    ts = str(peak.get("timestamp") or "")
    base = r.get("baseline") or {}
    at_peak = base.get("at_observed_peak_slot")
    increase = None
    if at_peak and peak.get("flow") is not None:
        increase = (float(peak["flow"]) - float(at_peak)) / float(at_peak) * 100.0
    extra = [f"Window:       {r.get('hour_from')}:00–{r.get('hour_to')}:00, "
             f"total {_fmt(r.get('total'))}"
             + (f" ({_signed_pct(base.get('change_pct'))} vs. baseline)"
                if base.get("change_pct") is not None else "")]
    if r.get("is_closed"):
        extra.append("Note:         station partly closed in this window (zero values)")
    if r.get("capped_slots"):
        extra.append(f"Note:         {r['capped_slots']} slot(s) at the cap – peak censored")
    weekday = ""
    if base.get("method"):
        weekday = next((d for d in WEEKDAYS if d in base["method"]), "")
    return _peak_box(
        r.get("station_name"), ts[11:16] or "–", peak.get("flow"), date=r.get("date"),
        baseline=at_peak if at_peak is not None else None,
        baseline_label=f"mean of {weekday}s, same slot" if weekday else "same slot",
        increase_pct=increase, extra=extra,
    )


WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _peak_15min(r: dict) -> list[str]:
    scope = "" if r.get("scope") == "single day" else "maximum across all data days"
    extra = [f"Note:         {scope}"] if scope else []
    if r.get("phase"):
        extra.insert(0, f"Phase:        {r['phase']} ({r.get('hour_from')}:00–{r.get('hour_to')}:00)")
    if r.get("distance_m") is not None:
        extra.append(f"Walking dist: {_fmt(r['distance_m'])} m to venue (straight-line approx.)")
    if r.get("slots_at_own_max", 0) > 1:
        extra.append(f"Note:         {r['slots_at_own_max']} slot(s) at station max")
    return _peak_box(r.get("station_name"), r.get("peak_time"), r.get("peak_value"),
                     station_max_ever=r.get("station_max_ever"),
                     pct_of_own_max=r.get("pct_of_own_max"),
                     capacity_status=r.get("capacity_status"),
                     date=r.get("date"), extra=extra)


def _peak_vs_baseline(r: dict) -> list[str]:
    weekday = r.get("weekday") or ""
    label = (f"dry {weekday}" if r.get("baseline_dry_only")
             else f"{weekday}, all weather")
    return _peak_box(
        r.get("station_name"), r.get("peak_time"), r.get("peak_value"), date=r.get("date"),
        station_max_ever=r.get("station_max_ever"),
        pct_of_own_max=r.get("pct_of_own_max"),
        capacity_status=r.get("capacity_status"),
        baseline=r.get("baseline_value"),
        baseline_label=f"{label}, median of {r.get('baseline_days')} days",
        increase_pct=r.get("increase_pct"),
        confounder=r.get("confounder_detail"),
    )


def _typical_load(r: dict) -> list[str]:
    return [
        f"**{_short(r.get('station_name'))}**: typical weekday peak "
        f"{_fmt(r.get('typical_peak_mean'), 1)} at {r.get('typical_peak_time')} = "
        f"{_ampel(r.get('capacity_status'))} ({_fmt(r.get('pct_of_own_max'), 1)}% of "
        f"station max {_fmt(r.get('station_max_ever'))}, reached {r.get('station_max_at')})."
    ]


def _network_summary(r: dict) -> list[str]:
    peak = r.get("peak_slot") or {}
    lines = [
        f"Network on {r.get('date')}: **{_num(r.get('total_network_flow'))}** "
        f"passengers, peak slot {_num(peak.get('total_flow'))} at {peak.get('timestamp')} "
        f"(mean per slot {_num(r.get('mean_per_slot'))})."
    ]
    top = r.get("top_5_stations") or []
    if top:
        lines.append("Busiest stations: " + ", ".join(
            f"{_short(s['station_name'])} ({_num(s['total'])})" for s in top[:MAX_ITEMS]
        ))
    return lines


def _anomalies(r: dict) -> list[str]:
    items = r.get("anomalies") or []
    lines = [f"{r.get('anomaly_count', 0)} station anomalies on {r.get('date')} "
             f"(|z| ≥ {r.get('z_threshold')} vs. other {r.get('weekday')}s)."]
    for a in items[:MAX_ITEMS]:
        lines.append(
            f"• {_short(a['station_name'])}: {_num(a['observed_total'])} vs. "
            f"baseline {_num(a['baseline_mean'])} (z = {a['z_score']}, {a['direction']})"
        )
    return lines


def _closures(r: dict) -> list[str]:
    lines = []
    if r.get("note"):
        lines.append(r["note"])
    if r.get("message"):
        lines.append(r["message"])
    items = r.get("closures") or []
    if items:
        lines.append(f"{r.get('count', len(items))} closure(s):")
        for c in items[:MAX_ITEMS]:
            lines.append(f"• {c['when']} – {c['end_time']}: {c['description']}")
        if len(items) > MAX_ITEMS:
            lines.append(f"… and {len(items) - MAX_ITEMS} more.")
    return lines


def _closure_impact(r: dict) -> list[str]:
    lines = [f"Closure: {r.get('closure')} ({r.get('when')} – {r.get('end_time')})."]
    for s in (r.get("neighboring_stations") or [])[:3]:
        lines.append(
            f"• Neighbour {_short(s['station'])}: {_num(s['flow_during'])} during "
            f"closure vs. baseline {_num(s['baseline'])} ({s.get('change_pct')} %)"
        )
    return lines


def _events(r: dict) -> list[str]:
    lines = []
    if r.get("is_future_event"):
        lines.append(
            f"{r.get('requested_date')} is after the last data day "
            f"({r.get('last_data_day')}) – historical venue patterns:"
        )
        for p in (r.get("historical_patterns") or [])[:3]:
            lines.append(
                f"• {p['venue_key']}: {p['past_events']} past events, nearest "
                f"{_short(p['nearest_station'])} ({_num(p['distance_m'])} m), peak "
                f"uplift {p.get('peak_uplift_pct')} %"
            )
        return lines
    items = r.get("events") or []
    lines.append(f"{r.get('count', len(items))} event(s):")
    for e in items[:MAX_ITEMS]:
        station = (f", nearest {_short(e['nearest_station'])} ({_num(e['distance_m'])} m)"
                   if e.get("nearest_station") else "")
        lines.append(
            f"• {e['event_name']} – {e['began_local']} at {e.get('venue') or e.get('address')}"
            f" (scaled attendance {_num(e.get('estimated_attendance'))}){station}"
        )
    return lines


def _weather(r: dict) -> list[str]:
    s = r.get("summary") or {}
    return [
        f"Weather {r.get('date')}: {s.get('conditions')}, mean {s.get('temp_mean')} °C "
        f"(max {s.get('temp_max')} °C), rain {s.get('prcp_total')} mm "
        f"(peak rate {s.get('max_prcp_rate_mm_per_h')} mm/h)."
    ]


def _weather_anomaly(r: dict) -> list[str]:
    peak = r.get("peak") or {}
    return [
        f"Strongest {r.get('metric')} in week from {r.get('week_start')}: "
        f"{peak.get('value')} at {peak.get('timestamp')} ({r.get('conditions_at_peak')}, "
        f"z = {r.get('z_score')})."
    ]


def _energy(r: dict) -> list[str]:
    # MWh pro 1.000 Fahrgaeste = kWh pro Fahrgast. Verhaeltnis mit 4
    # Nachkommastellen (U5/U6 liegen nur 1,3 % auseinander), Fahrgaeste ganz,
    # Prozente mit 2 Stellen.
    lines = [f"Worst efficiency: **{r.get('worst_efficiency_line')}**, best: "
             f"**{r.get('best_efficiency_line')}** (kWh per passenger = MWh per 1,000 passengers)."]
    ranking = r.get("efficiency_ranking") or []
    # Relativer Vergleich: jede Linie gegen das Mittel aller Linien.
    mean = (sum(e["mwh_per_1000_pax"] for e in ranking) / len(ranking)) if ranking else None
    for e in ranking[:MAX_ITEMS]:
        rel = (f", {(e['mwh_per_1000_pax'] - mean) / mean * 100.0:+.2f}% vs. line mean"
               if mean else "")
        pax = f", {e['total_pax']:,.0f} pax" if e.get("total_pax") is not None else ""
        lines.append(f"• {e['line']}: {e['mwh_per_1000_pax']:.4f} kWh/pax "
                     f"({e['total_mwh']:,.0f} MWh total{pax}{rel})")
    return lines


def _critical(r: dict) -> list[str]:
    lines = ["Most critical stations (stations cut off if closed):"]
    for s in (r.get("top_critical_stations") or [])[:MAX_ITEMS]:
        lines.append(
            f"{s['rank']}. {_short(s['station_name'])} – cuts off "
            f"{s['stations_cut_off']} stations, ~{_num(s['affected_passengers_estimate'])} "
            "passengers/day affected"
        )
    return lines


def _transit_legs(t: dict) -> list[str]:
    """Abschnitte einer multimodalen Route (find_transit_route)."""
    route = (t.get("routes") or [{}])[0]
    lines = [f"{_num(t.get('total_minutes'), 1)} min, {t.get('transfers')} transfer(s):"]
    for leg in route.get("legs") or []:
        lines.append(f"• {leg['line']} ({leg['mode']}): {_short(leg['from'])} → "
                     f"{_short(leg['to'])}, {leg['stops']} stops, {leg['minutes']} min")
    return lines


def _routes(r: dict) -> list[str]:
    # Station ausserhalb des U-Bahn-Datensatzes: Ergebnis kommt aus dem Gesamtnetz.
    if r.get("fallback_reason"):
        return [r["fallback_reason"]] + _transit_legs(r)
    lines = []
    if r.get("premise_warning"):
        lines.append(f"Note: {r['premise_warning']}")
    if r.get("suspended_section"):
        lines.append("Suspended section: " + " – ".join(
            _short(s) for s in r["suspended_section"]))
    routes = r.get("routes") or []
    if not routes:
        lines.append(r.get("status") or r.get("detail") or "No U-Bahn route found.")
        if r.get("tip"):
            lines.append(r["tip"])
    for route in routes[:3]:
        lines.append(f"• {route['stops']} stops: " + " → ".join(
            _short(s) for s in route["path"]))
    if r.get("transit_alternative"):
        lines.append("Multimodal alternative (bus/tram/S-Bahn):")
        lines += _transit_legs(r["transit_alternative"])
    return lines


def _transit_route(r: dict) -> list[str]:
    return [f"{_short(r.get('from'))} → {_short(r.get('to'))}:"] + _transit_legs(r)


def _temporal_context(r: dict) -> list[str]:
    return [f"{r.get('timestamp')} ({r.get('weekday')}): slot **{r.get('time_slot')}**, "
            f"rush hour: {r.get('ist_rushhour')}, weekend: {r.get('ist_wochenende')}, "
            f"service running: {r.get('service_running')}."]


def _peak_hours(r: dict) -> list[str]:
    m, e, p = r.get("morning_peak") or {}, r.get("evening_peak") or {}, r.get("service_pause") or {}
    return [f"Peak hours Mon–Fri {m.get('start')}–{m.get('end')} and {e.get('start')}–"
            f"{e.get('end')}; no peak hours at weekends. Service pause "
            f"{p.get('start')}–{p.get('end')}."]


def _peak_profile(r: dict) -> list[str]:
    m, e = r.get("morning_peak") or {}, r.get("evening_peak") or {}
    return [
        f"{_short(r.get('station_name'))} ({r.get('day_selection')}, "
        f"{r.get('days_included')} days): morning peak {m.get('time')} "
        f"({_num(m.get('mean_flow'), 1)}), evening peak {e.get('time')} "
        f"({_num(e.get('mean_flow'), 1)})."
    ]


def _compare(r: dict) -> list[str]:
    return [
        f"{_short(r.get('station_name'))}: morning peak {_num(r.get('station_morning_peak_mean'), 1)} "
        f"vs. network mean {_num(r.get('network_morning_peak_mean'), 1)} "
        f"({r.get('difference_pct')} %), rank {r.get('rank_among_stations')} of "
        f"{r.get('total_stations')}."
    ]


def _weekday_profile(r: dict) -> list[str]:
    if r.get("weekday") and r.get("hourly_profile"):
        m, e = r.get("morning_peak") or {}, r.get("evening_peak") or {}
        return [f"{r.get('scope')} on {r['weekday']}s ({r.get('days_included')} days): "
                f"{_num(r.get('mean_daily_total'))} per day, morning peak {m.get('time')} "
                f"({_num(m.get('mean_flow'))}), evening peak {e.get('time')} "
                f"({_num(e.get('mean_flow'))})."]
    lines = [f"{r.get('scope')}: busiest {r.get('busiest_weekday')}, quietest "
             f"{r.get('quietest_weekday')}."]
    for d in (r.get("weekday_ranking") or [])[:7]:
        lines.append(f"• {d['weekday']}: {_num(d['mean_daily_total'])} per day")
    return lines


def _busiest_by_weekday(r: dict) -> list[str]:
    scope = "on all days" if r.get("weekday") == "all days" else f"on {r.get('weekday')}s"
    lines = [f"Busiest stations {scope} ({r.get('time_window')}, "
             f"{r.get('days_included')} days):"]
    for s in (r.get("top_5_stations") or [])[:MAX_ITEMS]:
        total = s.get("mean_daily_total", s.get("mean_window_total"))
        served = f" [{s['lines']}]" if s.get("lines") else ""
        lines.append(f"{s['rank']}. {_short(s['station_name'])}{served} – "
                     f"{_num(total)} on average, peak {s['peak_time']}")
    return lines


def _threshold(r: dict) -> list[str]:
    unit = "hour" if "hour" in str(r.get("unit")) else "15-min slot"
    lines = [f"{r.get('count')} of {r.get('total_stations')} stations exceed "
             f"{_num(r.get('threshold'))} passengers per {unit} on average "
             f"({r.get('day_selection')}):"]
    key = "peak_mean_per_hour" if unit == "hour" else "peak_mean_per_slot"
    for s in (r.get("stations") or [])[:MAX_ITEMS * 2]:
        lines.append(f"• {_short(s['station_name'])}: peak {_num(s.get(key))} "
                     f"at {s['peak_time']}, above threshold {s['first_time_above']}–{s['last_time_above']}")
    return lines


def _rain_impact(r: dict) -> list[str]:
    s = r.get("summary") or {}
    lines = [f"Rainy hours ({s.get('rain_hours')}) vs. comparable dry hours: "
             f"**{s.get('mean_change_rain_pct')} %** network flow on average ({r.get('period')})."]
    for name, c in (r.get("by_intensity") or {}).items():
        lines.append(f"• {name} rain ({c['mm_per_h']} mm/h): {c['hours']} hours, "
                     f"{c['mean_change_pct']} %")
    return lines


def _weekly_summary(r: dict) -> list[str]:
    peak = r.get("peak_day") or {}
    lines = [f"Week {r.get('week')} ({r.get('week_type')}): **{_num(r.get('total_weekly_pax'))}** "
             f"station entries in total, peak {peak.get('weekday')} {peak.get('date')} "
             f"({_num(peak.get('pax'))})."]
    for day, v in (r.get("daily_totals") or {}).items():
        lines.append(f"• {day} ({v['weekday']}): {_num(v['total_pax'])}")
    return lines


def _station_weekday(r: dict) -> list[str]:
    return [f"{_short(r.get('station_name'))}: busiest {r.get('busiest_weekday')}, "
            f"quietest {r.get('quietest_weekday')}, pattern {r.get('pattern')} "
            f"(weekend/weekday {r.get('weekend_vs_weekday_ratio')})."]


def _network_info(r: dict) -> list[str]:
    return [f"{r.get('total_stations')} stations, {r.get('total_lines')} lines "
            f"({', '.join(r.get('lines') or [])}); {r.get('note')}. "
            f"Data period {r.get('data_period')}."]


def _gtfs_lines(r: dict) -> list[str]:
    lines = [f"Lines at {r.get('station')} (VBB timetable):"]
    for mode, items in (r.get("lines_by_type") or {}).items():
        lines.append(f"• {mode}: {', '.join(items)}")
    return lines


def _gtfs_stops(r: dict) -> list[str]:
    stops = r.get("stops") or []
    return [f"{r.get('line')} ({r.get('type')}, {r.get('stop_count')} stops, "
            f"{r.get('end_to_end_minutes')} min end to end): " + " → ".join(stops)]


def _gtfs_ubahn(r: dict) -> list[str]:
    return [f"U-Bahn lines in the VBB timetable: {', '.join(r.get('ubahn_lines') or [])}. "
            f"{r.get('note')}"]


def _gtfs_between(r: dict) -> list[str]:
    direct = r.get("direct_lines") or []
    return [f"Direct lines {r.get('from')} → {r.get('to')}: "
            + (", ".join(direct) if direct else "none (transfer needed)")]


def _dependencies(r: dict) -> list[str]:
    lines = [r.get("finding") or "Station dependencies:"]
    for p in (r.get("pairs") or [])[:MAX_ITEMS]:
        lines.append(f"• {_short(p['station_a'])} ↔ {_short(p['station_b'])}: r = "
                     f"{p['correlation']}, {p['graph_hops']} stops apart")
    return lines


def _disruption_routing(r: dict) -> list[str]:
    s = r.get("summary") or {}
    return [
        f"{s.get('segment_closures_analysed')} line-section closures analysed. Nearby "
        f"stations rose more than distant ones in {s.get('closures_where_nearby_stations_rose_more_than_distant')} "
        f"closures (mean {s.get('mean_uplift_near_sections_pct')} % vs. "
        f"{s.get('mean_uplift_far_away_pct')} %).",
        f"Only {s.get('share_on_shortest_detour_pct')} % of the stations with the largest "
        "increase lie on the shortest U-Bahn detour.",
    ]


def _diverse_routes(r: dict) -> list[str]:
    lines = [f"{_short(r.get('from'))} → {_short(r.get('to'))}:"]
    for route in (r.get("routes") or [])[:3]:
        lines.append(f"{route['rank']}. {route['summary']} – {route['total_with_transfers_minutes']} min "
                     f"(+{route['extra_minutes_vs_fastest']} min)")
    return lines


def _hotspot(r: dict) -> list[str]:
    lines = [f"**{r.get('hotspot')}**: {r.get('primary_problem')}"]
    if r.get("travel_phase") or r.get("direction"):
        phase = (r.get("travel_phase") or "travel phase open").capitalize()
        lines.append(f"{phase}, direction {r.get('direction') or 'open'}.")
    lines += ["", "| Alternative Station | Direction | Walk (min) | Notes |",
              "|---|---|---:|---|"]
    for a in r.get("alternatives") or []:
        walk = str(a["walking_time_mins"]) if a.get("walking_time_mins") else "–"
        lines.append(f"| {a['target_station']} | {a['direction_focus']} | {walk} | {a['note']} |")
    # Quellenhinweis einmal unter der Tabelle statt in jeder Zelle.
    lines += ["", "_Source: Hotspot planning values (data/hotspots.json) — no measured data._"]
    return lines


FORMATTERS: dict[str, Callable[[dict], list[str]]] = {
    "hotspot_alternatives": _hotspot,
    "find_station_dependencies": _dependencies,
    "analyze_disruption_routing": _disruption_routing,
    "find_diverse_transit_routes": _diverse_routes,
    "get_station_flow": _station_flow,
    "get_station_peak_15min": _peak_15min,
    "get_station_peak_vs_baseline": _peak_vs_baseline,
    "typical_load_status": _typical_load,
    "get_network_flow_summary": _network_summary,
    "detect_anomalies": _anomalies,
    "get_closures": _closures,
    "get_closure_impact": _closure_impact,
    "get_events": _events,
    "get_weather": _weather,
    "find_weather_anomalies": _weather_anomaly,
    "get_energy": _energy,
    "get_critical_stations": _critical,
    "find_alternative_routes": _routes,
    "get_peak_profile": _peak_profile,
    "compare_station_to_network": _compare,
    "get_weekday_profile": _weekday_profile,
    "get_busiest_station_by_weekday": _busiest_by_weekday,
    "get_station_weekday_pattern": _station_weekday,
    "get_weekly_summary": _weekly_summary,
    "network_info": _network_info,
    "get_lines_for_station": _gtfs_lines,
    "get_stops_for_line": _gtfs_stops,
    "get_all_ubahn_lines": _gtfs_ubahn,
    "find_route_between_stations": _gtfs_between,
    "find_transit_route": _transit_route,
    "find_stations_above_threshold": _threshold,
    "get_rain_impact": _rain_impact,
    "get_temporal_context": _temporal_context,
    "get_peak_hours": _peak_hours,
}


def _generic(r: dict) -> list[str]:
    """Skalare Top-Level-Werte eines unbekannten Tools."""
    pairs = [f"{k}: {v}" for k, v in r.items()
             if isinstance(v, (int, float, str)) and k != "data_limitation"]
    return ["; ".join(pairs[:6])] if pairs else []


# Überschriften in Betriebssprache statt Tool-Namen (Antwortformat: keine
# internen Namen in der Antwort).
TITLES: dict[str, str] = {
    "hotspot_alternatives": "Hotspot alternatives (planning values)",
    "find_station_dependencies": "Demand dependency between stations",
    "analyze_disruption_routing": "Passenger shifts during closures",
    "find_diverse_transit_routes": "Alternative corridors (bus/tram/S-Bahn/U-Bahn)",
    "get_temporal_context": "Time context",
    "get_peak_hours": "Peak hour definition",
    "find_stations_above_threshold": "Stations above the threshold",
    "get_rain_impact": "Rain and passenger numbers",
    "find_transit_route": "Multimodal route (bus/tram/S-Bahn/U-Bahn)",
    "get_station_flow": "Station flow",
    "get_station_peak_15min": "Peak load (15 min)",
    "get_station_peak_vs_baseline": "Peak vs. normal day",
    "typical_load_status": "Typical load status",
    "get_network_flow_summary": "Network flow",
    "detect_anomalies": "Unusual station counts",
    "get_closures": "Closures",
    "get_closure_impact": "Closure impact on neighbouring stations",
    "get_events": "Events",
    "get_weather": "Weather",
    "find_weather_anomalies": "Weather extremes",
    "get_energy": "Energy efficiency",
    "get_critical_stations": "Critical stations",
    "find_alternative_routes": "Alternative U-Bahn routes",
    "get_peak_profile": "Typical daily peaks",
    "compare_station_to_network": "Station vs. network peak",
    "get_weekday_profile": "Weekday profile",
    "get_busiest_station_by_weekday": "Busiest stations by weekday",
    "get_station_weekday_pattern": "Weekly pattern",
    "get_weekly_summary": "Weekly totals",
    "network_info": "Network overview",
    "get_lines_for_station": "Lines at the station (VBB timetable)",
    "get_stops_for_line": "Line stops (VBB timetable)",
    "get_all_ubahn_lines": "U-Bahn lines (VBB timetable)",
    "find_route_between_stations": "Direct connections (VBB timetable)",
}


def build_fallback_answer(gathered: dict[str, Any], assumptions: list[str],
                          reason: str) -> str:
    """Markdown-Antwort aus allen erfolgreichen und gescheiterten Tool-Ergebnissen."""
    # Daten zuerst: der Hinweis auf die LLM-freie Antwort steht am Ende.
    # Keine Markdown-Ueberschriften – Abschnitte als **Label:** (Antwortformat).
    parts: list[str] = []
    for entry in gathered["results"]:
        formatter = FORMATTERS.get(entry["name"], _generic)
        try:
            body = formatter(entry["result"])
        except (KeyError, TypeError, ValueError):
            body = _generic(entry["result"])
        if body:
            parts.append(f"**{TITLES.get(entry['name'], 'Data')}:**")
            parts.extend(body)
            parts.append("")
    missing = []
    for entry in gathered["failures"]:
        result = entry["result"]
        missing.append(f"{TITLES.get(entry['name'], 'Data')}: "
                       f"{result.get('detail') or result.get('error')}")
    if missing or assumptions:
        parts.append("**Limitations:**")
        parts.extend(missing)
        if assumptions:
            parts.append("Assumptions: " + " ".join(assumptions))
        parts.append("")
    # reason (Timeout/LLM-Ausfall) steht im Response-Feld "note", nicht im Text.
    parts.append("_Direct data answer — values taken unfiltered from operational data._")
    return "\n".join(parts).strip()
