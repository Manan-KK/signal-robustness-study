"""Dependency-free HTML viewer for aggregate sweep results."""

from __future__ import annotations

from html import escape
import json

import numpy as np
import pandas as pd


VIEWER_COLUMNS = (
    "fast_window",
    "slow_window",
    "observations",
    "brier_skill",
    "roc_auc",
)


def render_viewer(
    sweep: pd.DataFrame,
    *,
    title: str = "Synthetic signal robustness surface",
) -> str:
    """Render a self-contained viewer containing aggregate metrics only."""

    frame = _validate_sweep(sweep)
    records = frame.loc[:, VIEWER_COLUMNS].to_dict("records")
    payload = json.dumps(records, separators=(",", ":"), allow_nan=False).replace(
        "</", "<\\/"
    )
    safe_title = escape(title, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
:root{{--ink:#18212f;--muted:#5b6574;--line:#cbd3dd;--panel:#f5f7fa}}
*{{box-sizing:border-box}} body{{margin:0;font:15px system-ui,sans-serif;color:var(--ink);background:white}}
main{{max-width:980px;margin:36px auto;padding:0 22px}} h1{{font-size:27px;margin:0 0 8px}}
p{{color:var(--muted);line-height:1.5}} .controls{{display:flex;gap:14px;align-items:center;margin:22px 0}}
select{{padding:7px 10px;border:1px solid var(--line);border-radius:6px;background:white}}
table{{border-collapse:collapse;width:100%;table-layout:fixed}} th,td{{border:1px solid var(--line);padding:11px;text-align:center}}
th{{background:var(--panel);font-weight:650}} td.metric{{cursor:pointer;font-variant-numeric:tabular-nums}}
td.metric:focus{{outline:3px solid #2367d1;outline-offset:-3px}} #detail{{min-height:48px;padding:12px 14px;background:var(--panel);border-radius:7px;margin-top:18px}}
.note{{font-size:13px}} code{{color:var(--ink)}}
</style>
</head>
<body>
<main>
<h1>{safe_title}</h1>
<p>Every cell is an aggregate score from the declared synthetic window grid. Select a metric and click a cell for its complete summary.</p>
<div class="controls"><label for="metric">Metric</label><select id="metric"><option value="brier_skill">Brier skill</option><option value="roc_auc">ROC AUC</option></select></div>
<div id="matrix"></div><div id="detail" aria-live="polite">Select a cell.</div>
<p class="note">Brier skill is relative to a history-only base-rate forecast. This viewer contains no prices, dates, paths, or observation-level forecasts.</p>
</main>
<script>
const rows={payload};
const fast=[...new Set(rows.map(row=>row.fast_window))].sort((a,b)=>a-b);
const slow=[...new Set(rows.map(row=>row.slow_window))].sort((a,b)=>a-b);
const byPair=new Map(rows.map(row=>[`${{row.fast_window}}:${{row.slow_window}}`,row]));
const metric=document.getElementById('metric');
const matrix=document.getElementById('matrix');
const detail=document.getElementById('detail');
function color(value,name){{
  const centered=name==='roc_auc'?value-.5:value;
  const strength=Math.min(Math.abs(centered)*4,1);
  return centered>=0?`rgba(26,140,85,${{.12+.55*strength}})`:`rgba(196,55,55,${{.12+.55*strength}})`;
}}
function format(value){{return Number(value).toFixed(4)}}
function render(){{
  const name=metric.value;let html='<table><thead><tr><th>slow \\ fast</th>';
  fast.forEach(value=>html+=`<th>${{value}}</th>`);html+='</tr></thead><tbody>';
  slow.forEach(slowWindow=>{{html+=`<tr><th>${{slowWindow}}</th>`;fast.forEach(fastWindow=>{{
    const row=byPair.get(`${{fastWindow}}:${{slowWindow}}`);const value=row[name];
    html+=`<td class="metric" tabindex="0" data-fast="${{fastWindow}}" data-slow="${{slowWindow}}" style="background:${{color(value,name)}}">${{format(value)}}</td>`;
  }});html+='</tr>';}});html+='</tbody></table>';matrix.innerHTML=html;
  matrix.querySelectorAll('.metric').forEach(cell=>{{
    const show=()=>{{const row=byPair.get(`${{cell.dataset.fast}}:${{cell.dataset.slow}}`);detail.textContent=`fast=${{row.fast_window}}, slow=${{row.slow_window}}, observations=${{row.observations}}, Brier skill=${{format(row.brier_skill)}}, ROC AUC=${{format(row.roc_auc)}}`;}};
    cell.addEventListener('click',show);cell.addEventListener('keydown',event=>{{if(event.key==='Enter'||event.key===' '){{event.preventDefault();show();}}}});
  }});
}}
metric.addEventListener('change',render);render();
</script>
</body>
</html>
"""


def _validate_sweep(sweep: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(sweep, pd.DataFrame):
        raise TypeError("sweep must be a pandas DataFrame")
    missing = set(VIEWER_COLUMNS) - set(sweep.columns)
    if missing:
        raise ValueError(f"sweep is missing columns: {', '.join(sorted(missing))}")
    frame = sweep.loc[:, VIEWER_COLUMNS].copy()
    if frame.empty or frame.isna().any().any():
        raise ValueError("sweep must be nonempty and complete")
    if frame.duplicated(["fast_window", "slow_window"]).any():
        raise ValueError("sweep contains duplicate window pairs")
    numeric = frame.loc[:, VIEWER_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("sweep metrics must be finite")
    fast = sorted(frame["fast_window"].unique())
    slow = sorted(frame["slow_window"].unique())
    expected = {(int(x), int(y)) for x in fast for y in slow}
    actual = {
        (int(row.fast_window), int(row.slow_window))
        for row in frame.itertuples(index=False)
    }
    if actual != expected:
        raise ValueError("sweep must contain the complete rectangular grid")
    if (frame["observations"] <= 0).any():
        raise ValueError("observation counts must be positive")
    if (~frame["roc_auc"].between(0.0, 1.0)).any():
        raise ValueError("ROC AUC must be between zero and one")
    return frame.sort_values(["slow_window", "fast_window"]).reset_index(drop=True)
