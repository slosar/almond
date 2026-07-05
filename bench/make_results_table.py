#!/usr/bin/env python3
"""Render results/synth_n*_debugnode.json into a LaTeX table + console table."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = []
for f in sorted(root.glob("results/synth_n*_debugnode.json"),
                key=lambda p: int(p.stem.split("_")[1][1:])):
    d = json.loads(f.read_text())
    ducc = d.get("ducc", {})
    best_t = min(ducc.values()) if ducc else float("nan")
    best_nt = min(ducc, key=ducc.get) if ducc else "-"
    rows.append({
        "nside": d["nside"], "lmax": d["lmax"],
        "err": d["max_abs_rel_err_vs_ducc"],
        "alm_dev": d["alm_synth_device_s"],
        "alm_host": d["alm_synth_host_s"],
        "ducc1": ducc.get("1", float("nan")),
        "ducc_best": best_t, "ducc_best_nt": best_nt,
        "speed_dev": best_t / d["alm_synth_device_s"],
        "speed_host": best_t / d["alm_synth_host_s"],
        "pool_gb": d["alm_gpu_peak_pool_bytes"] / 1e9,
    })

hdr = (f"{'nside':>6} {'lmax':>6} {'acc':>9} {'almond-dev':>9} {'almond-host':>9} "
       f"{'ducc-1t':>9} {'ducc-best':>10} {'x-dev':>7} {'x-host':>7} {'GB':>5}")
print(hdr)
for r in rows:
    print(f"{r['nside']:>6} {r['lmax']:>6} {r['err']:>9.1e} "
          f"{r['alm_dev']:>9.4f} {r['alm_host']:>9.4f} {r['ducc1']:>9.3f} "
          f"{r['ducc_best']:>7.3f}@{r['ducc_best_nt']:<2} "
          f"{r['speed_dev']:>7.1f} {r['speed_host']:>7.1f} {r['pool_gb']:>5.2f}")

tex = [r"\begin{tabular}{rrcrrrrrr}", r"\toprule",
       r"$N_\mathrm{side}$ & $\ell_{\max}$ & acc.\ vs ducc & "
       r"Almond dev.\ [s] & Almond host [s] & ducc 1t [s] & ducc best [s] & "
       r"speedup & peak GB \\", r"\midrule"]
for r in rows:
    mant, expo = f"{r['err']:.0e}".split("e")
    acc = f"${mant}\\times10^{{{int(expo)}}}$"
    tex.append(
        f"{r['nside']} & {r['lmax']} & {acc}"
        f" & {r['alm_dev']:.4f} & {r['alm_host']:.4f} & {r['ducc1']:.3f}"
        f" & {r['ducc_best']:.3f} ({r['ducc_best_nt']}t)"
        f" & {r['speed_dev']:.1f}$\\times$ & {r['pool_gb']:.2f} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(root / "report" / "results_table.tex").write_text("\n".join(tex) + "\n")
print("\nwrote report/results_table.tex")
