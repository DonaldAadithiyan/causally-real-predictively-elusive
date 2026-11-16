"""Tables 2-4 of the camera-ready paper, regenerated from the committed result CSVs, in the
paper's own layout. Every cell is produced by the same functions that tests/test_paper_numbers.py
checks against the printed values, so the tables and the number tests cannot drift apart.

  Table 2 (tab:l2)     L2 evidence: held-out share of features whose U_j CI excludes 0; enrichment;
                       runs above the selection null.
  Table 3 (tab:exante) Ex-ante prediction of S_t (median Spearman rho, predictive runs).
  Table 4 (tab:taskg)  Continuous scorer minus naive_matched under shift.

Writes paper/tables/table{2,3,4}.{tex,md}.     python -m paper.make_tables
"""
from __future__ import annotations

import os

from mechconf.paths import REPO_ROOT
from paper import paper_numbers as pn

OUT = os.path.join(REPO_ROOT, "paper", "tables")
DS = ["ETTh1", "ETTh2", "electricity", "traffic"]


def _tex_num(s):
    return s.replace("-", "$-$").replace("+", "$+$")


def table2():
    rows = [("All pool features", [pn.l2_share(d, "all").replace("[", " [") for d in DS]),
            ("MVDF features", [pn.l2_share(d, "mvdf").replace("[", " [") for d in DS]),
            ("Enrichment (median)", [pn.enrichment_median(d).replace("times", "×") for d in DS]),
            ("Above selection null", [pn.above_selection_null(d) for d in DS])]
    return ["", *DS], rows


def table3():
    best = {"ETTh1": "F6", "ETTh2": "F2"}          # footnote: past errors + input/forecast stats (ETTh1); input stats (ETTh2)
    cell = lambda d, f: pn.exante_cell(d, f).replace("(", " (")
    rows = [("MVDF activations", [cell(d, "F0") for d in ("ETTh1", "ETTh2")]),
            ("Best ex-ante set†", [cell(d, best[d]) for d in ("ETTh1", "ETTh2")]),
            ("Dense hidden state", [cell(d, "F4") for d in ("ETTh1", "ETTh2")]),
            ("Full SAE code", [cell(d, "F5") for d in ("ETTh1", "ETTh2")])]
    return ["Feature set", "ETTh1", "ETTh2"], rows


def table4():
    cell = lambda d, m: pn.scorer_cell(d, m).replace("(", " (")
    rows = [("Dense", [cell(d, "dense") for d in DS]),
            ("MVDF", [cell(d, "mvdf") + ("*" if d == "electricity" else "") for d in DS])]
    return ["Input", *DS], rows


def write(name, header, rows):
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join([r[0], *(c.replace("-", "−") for c in r[1])]) + " |" for r in rows]
    open(os.path.join(OUT, f"{name}.md"), "w").write("\n".join(md) + "\n")
    esc = lambda s: s.replace("×", r"$\times$").replace("†", r"$^{\dagger}$").replace("*", r"$^{*}$")
    tex = [r"\begin{tabular}{|l|" + "c|" * (len(header) - 1) + "}", r"\hline",
           " & ".join(header) + r" \\", r"\hline"]
    tex += [" & ".join([esc(r[0]), *(esc(_tex_num(c)) for c in r[1])]) + r" \\" for r in rows]
    tex += [r"\hline", r"\end{tabular}"]
    open(os.path.join(OUT, f"{name}.tex"), "w").write("\n".join(tex) + "\n")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (("table2", table2), ("table3", table3), ("table4", table4)):
        write(name, *fn())
        print(f"wrote paper/tables/{name}.md and .tex")


if __name__ == "__main__":
    main()
