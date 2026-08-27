"""
Build a static HTML report (tables + layout similar to tb-metrics-eer-cm.canvas.tsx)
from metrics JSON files. Open the HTML in any browser — no Cursor Canvas required.

Phase I / II: <REDACTED_PATH> Test DB\\artifacts\\metrics_phase1.json, metrics_phase2.json
Phase III v1.01: artifacts\\metrics_phase3_version_1_01.json (val/test splits only)
Vector Balanced v1: artifacts\\metrics_phase3_vector_balanced_version_1.json (val/test)
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def table_binary_set(s: dict) -> str:
    cm = s["confusion_matrix"]["matrix"]
    tn, fp = cm[0]
    fn, tp = cm[1]
    eer = s["eer"]
    sup = s["support"]
    return f"""
    <section class="card">
      <h4 class="card-title">{html.escape(s["name"])}</h4>
      <p class="meta">N={s["n"]} · neg={sup["n0"]} · pos={sup["n1"]}</p>
      <table class="cm">
        <thead><tr><th></th><th>Pred 0</th><th>Pred 1</th></tr></thead>
        <tbody>
          <tr><th>True 0</th><td>{tn}</td><td>{fp}</td></tr>
          <tr><th>True 1</th><td>{fn}</td><td>{tp}</td></tr>
        </tbody>
      </table>
      <p class="meta">EER {pct(eer["eer"])} · thr {eer["threshold"]:.4f} · FPR {pct(eer["fpr"])} · FNR {pct(eer["fnr"])}</p>
    </section>
    """


def section_binary(title: str, model: str, thr: float, data: dict) -> str:
    parts = [f'<h2>{html.escape(title)}</h2>', f'<p class="meta">Model: {html.escape(model)} · CM threshold={thr}</p>', '<div class="grid-2">']
    for s in data["sets"]:
        parts.append(table_binary_set(s))
    parts.append("</div>")
    return "\n".join(parts)


def table_multiset(s: dict) -> str:
    rows = []
    for p in s["per_label"]:
        cm = p["confusion_matrix"]["matrix"]
        tn, fp = cm[0]
        fn, tp = cm[1]
        eer = p["eer"]
        sup = p["support"]
        rows.append(
            f"<tr><td>{html.escape(p['key'])}</td><td>{html.escape(p['name'])}</td>"
            f"<td>{sup['n0']}</td><td>{sup['n1']}</td>"
            f"<td>{pct(eer['eer'])}</td><td>{eer['threshold']:.4f}</td>"
            f"<td>{tn}</td><td>{fp}</td><td>{fn}</td><td>{tp}</td></tr>"
        )
    thead = "<tr><th>Key</th><th>Name</th><th>n0</th><th>n1</th><th>EER</th><th>thr@EER</th><th>TN</th><th>FP</th><th>FN</th><th>TP</th></tr>"
    return f"""
    <section class="card span-2">
      <h4 class="card-title">{html.escape(s["name"])}</h4>
      <p class="meta">N={s["n"]}</p>
      <table class="wide">
        <thead>{thead}</thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>
    """


def section_multi(title: str, model: str, thr: float, data: dict, split_filter: str | None) -> str:
    sets = data["sets"]
    if split_filter == "val_test":
        sets = [s for s in sets if "_val." in s["name"] or "_test." in s["name"] or s["name"].endswith("_val.npz") or s["name"].endswith("_test.npz")]
    parts = [f'<h2>{html.escape(title)}</h2>', f'<p class="meta">Model: {html.escape(model)} · CM threshold={thr}</p>', '<div class="grid-2">']
    for s in sets:
        parts.append(table_multiset(s))
    parts.append("</div>")
    return "\n".join(parts)


def build_html(phase1: dict, phase2: dict, p3: dict, vb: dict) -> str:
    blocks: list[str] = []
    if phase1:
        blocks.append(section_binary("Phase I (Normal vs Inactive+Active+NTM)", phase1["rf_model"], phase1["threshold"], phase1))
    if phase2:
        blocks.append(section_binary("Phase II (Inactive vs Active+NTM)", phase2["rf_model"], phase2["threshold"], phase2))
    if p3:
        note = (p3.get("dataset_title") or "Phase III Version 1.01") + " — " + (p3.get("dataset_note") or "active vs inactive (pre–Vector Balanced)")
        blocks.append(section_multi(note, p3["rf_model"], p3["threshold"], p3, "val_test"))
    if vb:
        blocks.append(
            section_multi(
                "Vector Balanced Version 1",
                vb["rf_model"],
                vb["threshold"],
                vb,
                "val_test",
            )
        )
    body = "\n".join(blocks)
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TB metrics — EER &amp; Confusion matrix (static report)</title>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 24px 32px; color: #1a1a1a; background: #fafafa; line-height: 1.45; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 8px; }}
    h2 {{ font-size: 1.15rem; margin-top: 28px; margin-bottom: 8px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
    h4.card-title {{ margin: 0 0 8px 0; font-size: 0.95rem; }}
    p.meta {{ font-size: 0.85rem; color: #444; margin: 4px 0 12px 0; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1100px; }}
    @media (max-width: 800px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    .card {{ background: #fff; border: 1px solid #ddd; padding: 14px 16px; }}
    .card.span-2 {{ grid-column: 1 / -1; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f0f0f0; }}
    footer {{ margin-top: 36px; font-size: 0.8rem; color: #555; max-width: 900px; }}
  </style>
</head>
<body>
  <h1>TB models — EER &amp; confusion matrix</h1>
  <p class="meta">Static report generated from metrics JSON (same numbers as Canvas data sources). EER from ROC; confusion matrices at fixed threshold.</p>
  {body}
  <footer>
    Cursor Canvas (<code>tb-metrics-eer-cm.canvas.tsx</code>) renders this content interactively inside the IDE.
    This HTML is for sharing, archiving, or printing outside Cursor.
  </footer>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "tb_metrics_eer_cm_report.html",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    tb_test = root.parent / "TB Test DB"
    phase1 = load_json(tb_test / "artifacts" / "metrics_phase1.json")
    phase2 = load_json(tb_test / "artifacts" / "metrics_phase2.json")
    p3 = load_json(root / "artifacts" / "metrics_phase3_version_1_01.json")
    vb = load_json(root / "artifacts" / "metrics_phase3_vector_balanced_version_1.json")
    html_out = build_html(phase1 or {}, phase2 or {}, p3 or {}, vb or {})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html_out, encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
