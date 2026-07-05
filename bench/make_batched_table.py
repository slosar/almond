#!/usr/bin/env python3
"""Render results/batch_n*_debugnode.json into console + LaTeX tables."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = []
for f in sorted(root.glob("results/batch_n*_debugnode.json"),
                key=lambda p: int(p.stem.split("_")[1][1:])):
    d = json.loads(f.read_text())
    rows.append(d)

print(f"{'nside':>6} {'B':>4} {'acc':>9} {'almond ms/col':>11} {'+copies':>9} "
      f"{'ducc64t ms/col':>15} {'x-dev':>6} {'x-host':>7}")
for d in rows:
    print(f"{d['nside']:>6} {d['batch']:>4} {d['max_abs_rel_err_vs_ducc']:>9.1e} "
          f"{d['alm_batch_device_per_col_s']*1e3:>11.2f} "
          f"{d['alm_batch_host_per_col_s']*1e3:>9.2f} "
          f"{d['ducc_batch_per_col_s']*1e3:>15.2f} "
          f"{d.get('speedup_device', float('nan')):>6.1f} "
          f"{d.get('speedup_host', float('nan')):>7.1f}")

tex = [r"\begin{tabular}{rrcrrrr}", r"\toprule",
       r"$N_\mathrm{side}$ & $B$ & acc. & Almond [ms/col] & Almond+copies & "
       r"ducc0 64t [ms/col] & speedup \\", r"\midrule"]
for d in rows:
    mant, expo = f"{d['max_abs_rel_err_vs_ducc']:.0e}".split("e")
    acc = f"${mant}\\times10^{{{int(expo)}}}$"
    tex.append(f"{d['nside']} & {d['batch']} & {acc}"
               f" & {d['alm_batch_device_per_col_s']*1e3:.2f}"
               f" & {d['alm_batch_host_per_col_s']*1e3:.2f}"
               f" & {d['ducc_batch_per_col_s']*1e3:.2f}"
               f" & {d.get('speedup_device', 0):.1f}$\\times$ \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(root / "report" / "batched_table.tex").write_text("\n".join(tex) + "\n")
print("\nwrote report/batched_table.tex")
