#!/usr/bin/env python3
"""
ubahn_ingest.py - inject new data files (e.g. the 22 Sep - 1 Oct evaluation set) into the data folder.

Each file is checked against the dataset schema before it is added:
    flows*        timestamps parse, 15-minute grain, every station column known, counts >= 0
    weather*      timestamps parse, temp and prcp present, 15-minute grain
    *events*      required columns, times parse, attendance numeric, every venue locatable
    closures*     when / duration / description parse, every description understood
    energy*       dates parse, line columns known, values numeric
Errors block the injection; warnings are reported (e.g. an unknown venue: add it to venues_extra.csv).

Usage
    python ubahn_ingest.py --data data/ new/*.csv            # validate, copy, refit, print status
    python ubahn_ingest.py --data data/ new/*.csv --dry-run  # validate only
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

import src.engine as ue
from src.engine import TOD_ORDER, UBahnEngine

KINDS = ('flows', 'weather', 'events', 'closures', 'energy')
TARGET_PREFIX = {'flows': 'flows', 'weather': 'weather_data', 'events': 'berlin_events', 'closures': 'closures', 'energy': 'energy_consumption'}


def detect_kind(path: str, header: list[str]) -> str | None:
    name = os.path.basename(path).lower()
    for k in ('flows', 'weather', 'closures', 'energy'):
        if name.startswith(k):
            return k
    if 'event' in name:
        return 'events'
    cols = {c.strip().lower() for c in header}
    if {'event_name', 'began_local'} <= cols:
        return 'events'
    if {'when', 'duration', 'description'} <= cols:
        return 'closures'
    if {'temp', 'prcp'} <= cols:
        return 'weather'
    if any(re.fullmatch(r'u\d', c) for c in cols):
        return 'energy'
    if len(header) > 100:
        return 'flows'
    return None


def _parse_ts(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, format='%m/%d/%Y %H:%M')
    except (ValueError, TypeError):
        return pd.to_datetime(s, format='mixed', errors='coerce')


def _grain(ts: pd.Series) -> list[str]:
    msgs = []
    ts = ts.dropna().sort_values()
    if len(ts) < 2:
        return ['fewer than two timestamps']
    step = ts.diff().dropna().mode().iloc[0]
    if step != pd.Timedelta('15min'):
        msgs.append(f'most common time step is {step}, expected 15 min')
    if ts.duplicated().any():
        msgs.append(f'{int(ts.duplicated().sum())} duplicate timestamps (the last one is kept)')
    service = pd.date_range(ts.min(), ts.max(), freq='15min')
    service = service[np.isin(service.strftime('%H:%M'), TOD_ORDER)]
    missing = service.difference(pd.DatetimeIndex(ts))
    if len(missing):
        msgs.append(f'{len(missing)} service slots missing, e.g. {missing[0]:%Y-%m-%d %H:%M}')
    return msgs


def validate(path: str, ref: UBahnEngine) -> dict:
    """Check one file. ref = an engine loaded (fit=False is enough) on the target data folder."""
    rep = {'file': os.path.basename(path), 'kind': None, 'rows': 0, 'range': None, 'errors': [], 'warnings': []}
    try:
        df = ue.read_csv_any(path)
    except Exception as ex:
        rep['errors'].append(f'cannot read CSV: {ex}')
        return rep
    kind = detect_kind(path, list(df.columns))
    if (kind == 'events' or (kind is None and df.shape[1] == len(ue.EVENT_COLUMNS))) and 'event_name' not in df.columns:
        df = ue.read_events_csv(path)            # headerless events file (official test set): first line is an event
        kind = 'events'
    rep['kind'], rep['rows'] = kind, len(df)
    if kind is None:
        rep['errors'].append('cannot tell which dataset this is (name it flows*, weather*, *events*, closures* or energy*)')
        return rep
    E, W = rep['errors'], rep['warnings']
    if kind == 'flows':
        ts = _parse_ts(df.iloc[:, 0])
        if ts.isna().any():
            E.append(f'{int(ts.isna().sum())} timestamps do not parse')
        W.extend(_grain(ts))
        names = [re.sub(r'\.\d+$', '', c) for c in df.columns[1:]]
        known = list(ref.stations.station_name)
        unknown = sorted(set(names) - set(known))
        missing = sorted(set(known) - set(names))
        if unknown:
            E.append(f'{len(unknown)} unknown station columns: {unknown[:5]}')
        if missing:
            W.append(f'{len(missing)} stations missing: {missing[:5]}')
        vals = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        if vals.isna().any().any():
            W.append(f'{int(vals.isna().sum().sum())} empty or non-numeric counts')
        if (vals < 0).any().any():
            E.append('negative passenger counts')
        rep['range'] = f'{ts.min():%Y-%m-%d %H:%M} -> {ts.max():%Y-%m-%d %H:%M}'
        over = pd.DatetimeIndex(ts.dropna()).intersection(ref.flows.index)
        if len(over):
            W.append(f'{len(over)} slots already loaded: the newer file will replace them')
    elif kind == 'weather':
        ts = _parse_ts(df.iloc[:, 0])
        if ts.isna().any():
            E.append(f'{int(ts.isna().sum())} timestamps do not parse')
        for c in ('temp', 'prcp'):
            if c not in df.columns:
                E.append(f'missing column {c}')
            elif df[c].isna().any():
                W.append(f'{int(df[c].isna().sum())} empty values in {c}')
        W.extend(_grain(ts))
        rep['range'] = f'{ts.min():%Y-%m-%d %H:%M} -> {ts.max():%Y-%m-%d %H:%M}'
    elif kind == 'events':
        need = ['event_name', 'began_local', 'estimated_end_local', 'venue_name', 'address', 'segment', 'estimated_attendance']
        miss = [c for c in need if c not in df.columns]
        if miss:
            E.append(f'missing columns {miss}')
            return rep
        st = pd.to_datetime(df.began_local, utc=True, errors='coerce')
        if st.isna().any():
            E.append(f'{int(st.isna().sum())} start times do not parse')
        if pd.to_numeric(df.estimated_attendance, errors='coerce').isna().any():
            E.append('non-numeric attendance')
        if df.estimated_end_local.isna().any():
            W.append(f'{int(df.estimated_end_local.isna().sum())} events without end time (start + 3 h for music, + 2 h otherwise)')
        lost = sorted({f'{a} ({v})' for a, v in zip(df.address.astype(str), df.venue_name.fillna(''))
                       if ref.geocode(str(a).strip(), v)[0] is None})
        if lost:
            W.append(f'{len(lost)} venues cannot be located, their events will have no effect until added to '
                     f'venues_extra.csv (name, lat, lon, aliases): {lost[:8]}')
        big = df.estimated_attendance.max()
        if big > 5000:
            W.append(f'attendance up to {int(big)}: far above training (max ~2,400 per listing); effects are scaled linearly')
        loc = st.dt.tz_convert('Europe/Berlin')
        rep['range'] = f'{loc.min():%Y-%m-%d} -> {loc.max():%Y-%m-%d}'
    elif kind == 'closures':
        miss = [c for c in ('when', 'duration', 'description') if c not in df.columns]
        if miss:
            E.append(f'missing columns {miss}')
            return rep
        start = _parse_ts(df['when'])
        if start.isna().any():
            E.append(f'{int(start.isna().sum())} start times do not parse')
        dur = df['duration'].map(ue._parse_duration)
        if (dur <= pd.Timedelta(0)).any():
            E.append(f"{int((dur <= pd.Timedelta(0)).sum())} durations do not parse (expected e.g. '3h30min')")
        for d in df['description']:
            c = ref.parse_closure(str(d))
            if c['kind'] == 'unknown' or c['issues']:
                W.append(f'not understood: "{d}" ({"; ".join(c["issues"])[:120]})')
        rep['range'] = f'{start.min():%Y-%m-%d} -> {start.max():%Y-%m-%d}'
    elif kind == 'energy':
        d = pd.to_datetime(df.iloc[:, 0], format='mixed', errors='coerce')
        if d.isna().any():
            E.append(f'{int(d.isna().sum())} dates do not parse')
        lines = [c for c in df.columns[1:]]
        bad = [c for c in lines if not re.fullmatch(r'U\d', str(c).strip())]
        if bad:
            E.append(f'unexpected columns {bad}')
        if df[lines].apply(pd.to_numeric, errors='coerce').isna().any().any():
            W.append('empty or non-numeric energy values')
        rep['range'] = f'{d.min():%Y-%m-%d} -> {d.max():%Y-%m-%d}'
    return rep


def ingest(files: list[str], data_dir: str, dry_run: bool = False, ref: UBahnEngine | None = None,
           learn: bool = True) -> dict:
    """Validate files and copy the valid ones into data_dir. Returns a report; nothing is copied if any
    file has errors. Copied files get their dataset prefix so the engine picks them up, and the newest
    file wins where timestamps overlap. With learn=True the engine is refitted on the updated folder and the
    report includes 'learning_report' (what the model learned, also saved as learning_report.json) and
    '_engine' (the refitted engine, for callers that want to reuse it)."""
    ref = ref or UBahnEngine(data_dir, fit=False)
    extra = [f for f in files if os.path.basename(f) == 'venues_extra.csv']
    if extra:                                                  # venues first, so the event check sees them
        for r in pd.read_csv(extra[0]).itertuples():
            aliases = [ue._ascii(a.strip()) for a in str(getattr(r, 'aliases', '') or '').split(';') if a.strip() and a.strip() != 'nan']
            ue.VENUES[str(r.name)] = (float(r.lat), float(r.lon), aliases + [ue._ascii(str(r.name))])
        if not dry_run:
            shutil.copy(extra[0], os.path.join(data_dir, 'venues_extra.csv'))
    reports = [validate(f, ref) for f in files if os.path.basename(f) != 'venues_extra.csv']
    ok = all(not r['errors'] for r in reports)
    before = None
    if ok and not dry_run and learn:
        import src.ops as O
        if not hasattr(ref, 'w'):
            ref.fit()
        before = O.engine_state(ref)
    copied = []
    if ok and not dry_run:
        for f, r in zip([f for f in files if os.path.basename(f) != 'venues_extra.csv'], reports):
            base = os.path.basename(f)
            prefix = TARGET_PREFIX[r['kind']]
            name = base if base.lower().startswith(prefix) else f'{prefix}_{base}'
            dest = os.path.join(data_dir, name)
            if os.path.exists(dest) and os.path.abspath(dest) != os.path.abspath(f):
                stem, ext = os.path.splitext(name)
                dest = os.path.join(data_dir, f'{stem}_{pd.Timestamp.now():%Y%m%d%H%M%S}{ext}')
            if os.path.abspath(dest) != os.path.abspath(f):
                shutil.copy(f, dest)
            os.utime(dest)                                     # newest file wins on overlapping timestamps
            copied.append(os.path.basename(dest))
    out = {'ok': ok, 'dry_run': dry_run, 'files': reports, 'copied': copied + (['venues_extra.csv'] if extra and not dry_run else [])}
    if before is not None:
        import src.ops as O
        new_eng = UBahnEngine(data_dir)
        out['learning_report'] = O.learning_report(before, new_eng, save_to=data_dir)
        out['_engine'] = new_eng
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('files', nargs='+')
    ap.add_argument('--data', required=True, help='data folder used by the engine')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    rep = ingest(a.files, a.data, a.dry_run)
    for r in rep['files']:
        flag = 'ERROR' if r['errors'] else ('warn ' if r['warnings'] else 'ok   ')
        print(f"{flag} {r['file']:45s} {str(r['kind']):8s} {r['rows']:6d} rows  {r['range']}")
        for m in r['errors']:
            print(f'        error: {m}')
        for m in r['warnings']:
            print(f'        warning: {m}')
    if not rep['ok']:
        print('\nNothing injected: fix the errors above.'); sys.exit(1)
    if a.dry_run:
        print('\nDry run: nothing copied.'); return
    print(f"\nCopied: {rep['copied']}")
    eng = rep.pop('_engine', None) or UBahnEngine(a.data)
    if 'learning_report' in rep:
        print('\nWhat the model learned (saved to learning_report.json):')
        for line in rep['learning_report']['summary']:
            print('  - ' + line)
    status = eng.data_status()
    print(json.dumps({k: status[k] for k in ('flows', 'weather', 'events', 'closures', 'energy', 'events_not_located', 'closures_not_understood')},
                     indent=1, ensure_ascii=False))
    new_start = max(eng.flows.index.max() - pd.Timedelta('9D'), eng.flows.index.min()).normalize() + pd.Timedelta('5h')
    print('Model check on the latest days (observed / expected, ~1.0 = behaves like training):')
    print(json.dumps(eng.calibration(new_start, eng.flows.index.max()), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
