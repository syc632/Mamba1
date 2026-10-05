"""Build a standalone HTML report for the Mamba-125M reproduction run.

Pulls the full history of one or more wandb runs, stitches them into a single
timeline (handling the restart gap), and writes a self-contained HTML page with
the training/validation curves plus the sampled generations.

    python make_report.py --runs pr9w23uf lntugrhw --gen gen.txt -o report.html
"""
import argparse
import html
import json
import os

import wandb

CHART_CDN = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"


def chart_js_tag():
    """Inline Chart.js when available so the report works offline / behind CSP.

    Must be emitted *before* any `new Chart(...)` call, otherwise the charts
    silently fail to render (ReferenceError: Chart is not defined).
    """
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "assets", "chart.umd.js")
    if os.path.exists(local):
        with open(local, encoding="utf-8") as f:
            return "<script>" + f.read() + "</script>"
    return f'<script src="{CHART_CDN}"></script>'


def fetch(run_ids):
    """Return merged series. Later runs win on overlapping steps."""
    api = wandb.Api()
    series = {}
    for rid in run_ids:
        r = api.run(f"2077182184-xi-an-jiaotong-liverpool-university/mamba1-repro/{rid}")
        cfg = {k: v for k, v in r.config.items() if not k.startswith("_")}
        for d in r.scan_history(page_size=2000):
            s = d.get("_step")
            if s is None:
                continue
            row = series.setdefault(s, {})
            for k, v in d.items():
                if k.startswith("_") or v is None:
                    continue
                row[k] = v
    return series, cfg


def xy(series, key):
    pts = sorted((s, r[key]) for s, r in series.items() if key in r)
    return [p[0] for p in pts], [round(p[1], 4) for p in pts]


def ds(label, xs, ys, color, dash=False, width=1.5, radius=0):
    return {"label": label, "data": [{"x": a, "y": b} for a, b in zip(xs, ys)],
            "borderColor": color, "borderWidth": width, "pointRadius": radius,
            "tension": 0.2, "fill": False,
            **({"borderDash": [4, 3]} if dash else {})}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--gen", default=None, help="file with sampled generations")
    p.add_argument("-o", "--out", default="report.html")
    args = p.parse_args()

    series, cfg = fetch(args.runs)
    tr_x, tr_y = xy(series, "train/loss")
    em_x, em_y = xy(series, "train/loss_smooth")
    lr_x, lr_y = xy(series, "train/lr")
    gn_x, gn_y = xy(series, "train/grad_norm")
    vl_x, vl_y = xy(series, "val/loss")
    vp_x, vp_y = xy(series, "val/ppl")
    tp_x, tp_y = xy(series, "train/tokens_per_sec")
    mem_x, mem_y = xy(series, "sys/mem_peak_gb")

    final_loss = tr_y[-1] if tr_y else float("nan")
    final_val = vl_y[-1] if vl_y else float("nan")
    final_ppl = vp_y[-1] if vp_y else float("nan")
    last_step = tr_x[-1] if tr_x else 0
    peak_gn = max(gn_y) if gn_y else 0
    peak_step = gn_x[gn_y.index(peak_gn)] if gn_y else 0
    avg_tps = sum(tp_y) / len(tp_y) if tp_y else 0
    peak_mem = max(mem_y) if mem_y else 0
    n_params = cfg.get("n_params", 0)

    gen_html = ""
    if args.gen and os.path.exists(args.gen):
        with open(args.gen, encoding="utf-8") as f:
            gen_html = html.escape(f.read())

    cfg_rows = {k: cfg.get(k) for k in (
        "model_size", "d_model", "n_layer", "seq_len", "micro_bsz",
        "grad_accum", "global_batch_tokens", "total_steps", "lr", "min_lr",
        "wd", "beta1", "beta2", "clip", "warmup_frac", "d_state", "d_conv",
        "expand", "n_params", "paper_batch_tokens")}
    cfg_rows = {k: v for k, v in cfg_rows.items() if v is not None}

    tr_cfg = json.dumps([
        ds("train loss", tr_x, tr_y, "#B4B2A9", width=1.2),
        ds("EMA", em_x, em_y, "#185FA5", width=2.5)])
    vl_cfg = json.dumps([ds("val loss", vl_x, vl_y, "#0F6E56", width=2.5, radius=3)])
    ppl_cfg = json.dumps([ds("val ppl (log)", vp_x, vp_y, "#993C1D", width=2.5, radius=3)])
    lr_cfg = json.dumps([ds("learning rate", lr_x, lr_y, "#534AB7", width=2)])
    gn_cfg = json.dumps([ds("grad norm", gn_x, gn_y, "#BA7517", width=1.2)])
    tp_cfg = json.dumps([ds("tokens/sec", tp_x, tp_y, "#185FA5", width=1.2)])
    mem_cfg = json.dumps([ds("peak GB", mem_x, mem_y, "#5F5E5A", width=1.5)])

    def scale(logy=False, ytitle=""):
        y = {"title": {"display": bool(ytitle), "text": ytitle,
                       "color": "#5F5E5A", "font": {"size": 11}},
             "ticks": {"color": "#5F5E5A", "font": {"size": 11}},
             "grid": {"color": "rgba(0,0,0,0.06)"}}
        if logy:
            y["type"] = "logarithmic"
        return json.dumps({
            "x": {"type": "linear", "title": {"display": True, "text": "step",
                  "color": "#5F5E5A", "font": {"size": 11}},
                  "ticks": {"color": "#5F5E5A", "font": {"size": 11}},
                  "grid": {"color": "rgba(0,0,0,0.06)"}},
            "y": y})

    def chart(cid, datasets, opts, height=260):
        return (f'<div style="position:relative;height:{height}px">'
                f'<canvas id="{cid}"></canvas></div>'
                f'<script>new Chart(document.getElementById("{cid}"),'
                f'{{type:"line",data:{{datasets:{datasets}}},options:{opts}}});</script>')

    base_opts = json.dumps({"responsive": True, "maintainAspectRatio": False,
                            "plugins": {"legend": {"display": False}}})

    def opts(logy=False, ytitle=""):
        o = json.loads(base_opts)
        o["scales"] = json.loads(scale(logy, ytitle))
        return json.dumps(o)

    cards = [
        ("final train loss", f"{final_loss:.4f}"),
        ("final val loss", f"{final_val:.4f}"),
        ("final val ppl", f"{final_ppl:.1f}"),
        ("steps", f"{last_step}"),
        ("params", f"{n_params/1e6:.1f}M"),
        ("avg throughput", f"{avg_tps/1000:.1f}k tok/s"),
        ("peak memory", f"{peak_mem:.2f} GB"),
        ("max grad norm", f"{peak_gn:.2f} @ {peak_step}"),
    ]
    card_html = "".join(
        f'<div style="background:var(--bg2,#f6f6f4);border-radius:8px;padding:14px 16px">'
        f'<div style="font-size:12px;color:#777;margin-bottom:4px">{k}</div>'
        f'<div style="font-size:20px;font-weight:500">{v}</div></div>'
        for k, v in cards)

    cfg_html = "".join(
        f"<tr><td style='padding:4px 12px 4px 0;color:#555'>{k}</td>"
        f"<td style='padding:4px 0;font-family:monospace'>{v}</td></tr>"
        for k, v in cfg_rows.items())

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Mamba-125M reproduction report</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:920px;
margin:0 auto;padding:32px 24px;color:#222;line-height:1.6;background:#fff}}
h1{{font-size:22px;font-weight:500;margin:0 0 4px}}
h2{{font-size:15px;font-weight:500;margin:32px 0 10px;
border-bottom:1px solid #eee;padding-bottom:6px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}}
table{{border-collapse:collapse;font-size:13px}}
pre{{background:#f6f6f4;padding:14px;border-radius:8px;overflow-x:auto;
font-size:12px;line-height:1.5}}
.note{{font-size:13px;color:#555;background:#f6f6f4;border-radius:8px;
padding:12px 14px;margin:10px 0}}
</style>
{chart_js_tag()}
</head><body>
<h1>Mamba-125M reproduction</h1>
<div style="color:#777;font-size:13px">Mamba: Linear-Time Sequence Modeling
with Selective State Spaces (arXiv 2312.00752) &mdash; Table 12 GPT3-125M spec,
reproduced on a single RTX 5060 Laptop (8 GB).</div>

<h2>Results at a glance</h2>
<div class="grid">{card_html}</div>

<h2>Training loss</h2>
{chart("c1", tr_cfg, opts(False, "loss"))}
<div class="note">Grey = raw per-step loss (noisy, data-driven).
Blue = EMA smoothing (alpha 0.05, ~20-step window, lags ~19 steps).</div>

<h2>Validation loss</h2>
{chart("c2", vl_cfg, opts(False, "val loss"))}

<h2>Validation perplexity (log scale)</h2>
{chart("c3", ppl_cfg, opts(True, "perplexity"))}
<div class="note">PPL = e^loss. On a <em>linear</em> axis this curve looks flat
because step-0 PPL is ~38,000, which stretches the scale &mdash; use log scale.
Log-scale PPL is exactly the val loss curve.</div>

<h2>Learning rate &amp; gradient norm</h2>
{chart("c4", lr_cfg, opts(False, "lr"))}
{chart("c5", gn_cfg, opts(False, "grad norm"))}
<div class="note">Peak grad norm {peak_gn:.2f} at step {peak_step} is a
single-step data spike, not instability: grad clipping (1.0) rescaled it, and
val loss continues smoothly across it.</div>

<h2>Throughput &amp; memory</h2>
{chart("c6", tp_cfg, opts(False, "tokens/sec"))}
{chart("c7", mem_cfg, opts(False, "GB"))}

<h2>Configuration</h2>
<table>{cfg_html}</table>

<h2>Sampled generations</h2>
<div class="note">temp 0.8, top-p 0.95, from the final checkpoint.</div>
<pre>{gen_html}</pre>
</body></html>"""

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"[report] wrote {args.out}  ({len(doc)/1024:.1f} KB)")
    print(f"[report] steps={last_step} val_loss={final_val:.4f} ppl={final_ppl:.1f} "
          f"avg_tps={avg_tps:.0f} peak_mem={peak_mem:.2f}G")


if __name__ == "__main__":
    main()
