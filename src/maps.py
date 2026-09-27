"""
ubahn_maps.py - animated network maps for the operator app (plotly, no map tiles needed, works offline).

    replay_frames(eng, date, t_from, t_to)       recorded flow vs normal, slot by slot (incident replay)
    forecast_frames(eng, date, weather, events, closures)
                                                 chance of reaching the ceiling, slot by slot (risk map)
    network_figure(eng, frames, mode)            plotly figure with a play button and a time slider

Each frame carries a caption with the weather, the events in their arrival/departure phase and the closures
in force at that moment, so the animation explains itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.engine import UBahnEngine, _to_dt

LINE_COLORS = {'U1': '#7DAD4C', 'U2': '#DA421E', 'U3': '#16683D', 'U4': '#F0D722', 'U5': '#7E5330',
               'U6': '#8C6DAB', 'U7': '#528DBA', 'U8': '#224F86', 'U9': '#F3791D'}
MIN15 = pd.Timedelta('15min')


def _caption(eng, t, evs):
    parts = [f'{t:%a %d %b %H:%M}']
    if t in eng.weather.index:
        w = eng.weather.loc[t]
        parts.append(f"{w.temp:.0f}°C" + (f", rain {w.prcp:.1f} mm/h" if w.prcp > 0 else ', dry'))
    ph = []
    for e in evs:
        s0, e0 = _to_dt(e['start']), _to_dt(e['end'])
        if s0 - pd.Timedelta('90min') <= t <= s0 + MIN15:
            ph.append(f"{e.get('name', 'event')[:28]} (arrivals)")
        elif e0 <= t <= e0 + pd.Timedelta('2h'):
            ph.append(f"{e.get('name', 'event')[:28]} (departures)")
    if ph:
        parts.append('Events: ' + '; '.join(ph[:3]))
    cl = [c['description'].split(' due to')[0] for c in eng._clist if c['start'] and c['start'] <= t < c['end']]
    if cl:
        parts.append('Closures: ' + '; '.join(cl[:2]))
    return ' · '.join(parts)


def replay_frames(eng: UBahnEngine, date: str, t_from: str = '05:00', t_to: str = '00:45'):
    """Recorded reading vs the normal level (no events, weather included) for every slot of a window."""
    day = pd.Timestamp(date).normalize()
    a = day + pd.Timedelta(hours=int(t_from[:2]), minutes=int(t_from[3:]))
    b = day + pd.Timedelta(hours=int(t_to[:2]), minutes=int(t_to[3:]))
    if b < a:
        b += pd.Timedelta('1D')
    idx = eng.slots(a, b)
    idx = idx[idx <= eng.flows.index.max()]
    if len(idx) == 0:
        raise ValueError('no recorded data in this window')
    obs = eng.flows.reindex(idx)[eng.keys].values
    lam0 = eng.expected(idx[0], idx[-1], events=[]).reindex(idx).values
    norm = eng.expected_reading(lam0, eng.ceiling.values[None, :])
    evs = eng._data_events(idx[0], idx[-1])
    frames = []
    for i, t in enumerate(idx):
        ratio = (obs[i] + 1) / (norm[i] + 1)
        val = np.clip(np.log2(ratio), -2, 3)
        frames.append({'slot': t, 'value': val, 'size': 6 + 18 * np.sqrt(np.clip(obs[i], 0, None) / 600),
                       'hover': [f"<b>{k}</b><br>{t:%H:%M}: {int(o)} recorded<br>normal about {int(n)}<br>ceiling {int(c)}"
                                 for k, o, n, c in zip(eng.keys, obs[i], norm[i], eng.ceiling.values)],
                       'caption': _caption(eng, t, evs),
                       'top': sorted(zip(eng.keys, obs[i] - norm[i], obs[i], norm[i]), key=lambda x: -x[1])[:5]})
    return frames


def forecast_frames(eng: UBahnEngine, date: str, weather: dict | None = None, extra_events: list | None = None,
                    extra_closures: list | None = None):
    """Chance of reaching the ceiling for every slot of a service day (recorded weather/events when available,
    plus scenario additions)."""
    import src.ops as O
    day = pd.Timestamp(date).normalize()
    S, E, in_data, weather, evs, cls, notes = O._scenario(eng, day, weather, extra_events, extra_closures)
    lam = eng.expected(S, E, weather=weather, events=evs, closures=cls)
    cap = eng.ceiling.values[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(lam.values > 0, np.exp(-cap / np.maximum(lam.values, 1e-9)), 0)
    er = eng.expected_reading(lam.values, cap)
    frames = []
    for i, t in enumerate(lam.index):
        frames.append({'slot': t, 'value': p[i], 'size': 6 + 18 * np.sqrt(np.clip(er[i], 0, None) / 600),
                       'hover': [f"<b>{k}</b><br>{t:%H:%M}: about {int(x)} expected<br>chance of reaching the ceiling {pp:.0%}<br>ceiling {int(c)}"
                                 for k, x, pp, c in zip(eng.keys, er[i], p[i], eng.ceiling.values)],
                       'caption': _caption_forecast(eng, t, evs, cls, weather),
                       'top': sorted(zip(eng.keys, p[i], er[i], eng.ceiling.values), key=lambda x: -x[1])[:5]})
    return frames, notes


def _caption_forecast(eng, t, evs, cls, weather):
    parts = [f'{t:%a %d %b %H:%M}']
    if weather:
        parts.append(f"scenario {weather.get('tmean')}°C" + (f", rain {weather.get('prcp')} mm/h" if weather.get('prcp') else ', dry'))
    elif t in eng.weather.index:
        w = eng.weather.loc[t]
        parts.append(f"{w.temp:.0f}°C" + (f", rain {w.prcp:.1f} mm/h" if w.prcp > 0 else ', dry'))
    ph = []
    for e in evs:
        s0, e0 = _to_dt(e['start']), _to_dt(e['end'])
        if s0 - pd.Timedelta('90min') <= t <= s0 + MIN15:
            ph.append(f"{e.get('name', 'event')[:28]} (arrivals)")
        elif e0 <= t <= e0 + pd.Timedelta('2h'):
            ph.append(f"{e.get('name', 'event')[:28]} (departures)")
    if ph:
        parts.append('Events: ' + '; '.join(ph[:3]))
    cl = [c['description'].split(' due to')[0] for c in cls if c['start'] and c['start'] <= t < c['end']]
    if cl:
        parts.append('Closures: ' + '; '.join(cl[:2]))
    return ' · '.join(parts)


def network_figure(eng: UBahnEngine, frames: list, mode: str = 'replay', height: int = 640) -> go.Figure:
    """mode 'replay': colour = recorded vs normal (blue below, grey normal, red above);
    mode 'risk': colour = chance of reaching the ceiling (green to red)."""
    fig = go.Figure()
    for L, seq in eng.line_seq.items():
        xs, ys = [], []
        for a, b in zip(seq[:-1], seq[1:]):
            xs += [eng.stations.at[a, 'longitude'], eng.stations.at[b, 'longitude'], None]
            ys += [eng.stations.at[a, 'latitude'], eng.stations.at[b, 'latitude'], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color=LINE_COLORS.get(L, '#999'), width=3),
                                 name=L, hoverinfo='skip', opacity=0.55))
    if mode == 'replay':
        cscale = [[0, '#2563eb'], [0.4, '#d1d5db'], [0.55, '#fca5a5'], [1, '#b91c1c']]
        cmin, cmax, ctitle = -2, 3, 'recorded<br>vs normal'
        ticks = dict(tickvals=[-2, -1, 0, 1, 2, 3], ticktext=['÷4', '÷2', 'normal', '×2', '×4', '×8'])
    else:
        cscale = [[0, '#16a34a'], [0.3, '#facc15'], [0.6, '#f97316'], [1, '#b91c1c']]
        cmin, cmax, ctitle = 0, 1, 'chance of<br>reaching<br>the ceiling'
        ticks = dict(tickvals=[0, 0.25, 0.5, 0.75, 1], ticktext=['0%', '25%', '50%', '75%', '100%'])
    f0 = frames[0]
    lon, lat = eng.stations.longitude.values, eng.stations.latitude.values
    fig.add_trace(go.Scatter(x=lon, y=lat, mode='markers', name='stations', hovertext=f0['hover'], hoverinfo='text',
                             marker=dict(color=f0['value'], size=f0['size'], colorscale=cscale, cmin=cmin, cmax=cmax,
                                         line=dict(color='#374151', width=0.6),
                                         colorbar=dict(title=ctitle, thickness=12, len=0.6, **ticks))))
    st_idx = len(fig.data) - 1
    fig.frames = [go.Frame(name=f['slot'].strftime('%H:%M'), traces=[st_idx],
                           data=[go.Scatter(hovertext=f['hover'], marker=dict(color=f['value'], size=f['size']))],
                           layout=go.Layout(title_text=f['caption'])) for f in frames]
    steps = [dict(method='animate', label=f['slot'].strftime('%H:%M'),
                  args=[[f['slot'].strftime('%H:%M')], dict(mode='immediate', frame=dict(duration=0, redraw=True), transition=dict(duration=0))])
             for f in frames]
    fig.update_layout(
        title=dict(text=f0['caption'], font=dict(size=13), x=0.01), height=height, margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor='x', scaleratio=1.64),
        updatemenus=[dict(type='buttons', direction='left', x=0.01, y=0.02, xanchor='left', yanchor='bottom', showactive=False,
                          buttons=[dict(label='▶ Play', method='animate',
                                        args=[None, dict(frame=dict(duration=450, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                                   dict(label='❚❚ Pause', method='animate',
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate', transition=dict(duration=0))])])],
        sliders=[dict(active=0, x=0.18, y=0.0, len=0.8, xanchor='left', yanchor='bottom', pad=dict(t=0, b=0),
                      currentvalue=dict(prefix='', visible=False), steps=steps)])
    return fig
