"""Generate Cursor canvas TSX with embedded Vector Balanced v1 metrics JSON.

Uses an explicit CanvasData type (no `satisfies` / deep `as const`) so the TS
language service does not choke on huge literal inference when opening the file.
"""

import json
from pathlib import Path

CANVAS_PATH = Path(r"C:\Users\SEJONG_ENDO_3\.cursor\projects\d\canvases\vector-balanced-version-1.canvas.tsx")
DATA_JSON = Path(__file__).resolve().parent / "artifacts" / "vector_balanced_version_1" / "canvas_embed_data.json"

TSX_HEAD = r"""import { Divider, Grid, H1, H2, H3, Row, Spacer, Stack, Text, useHostTheme } from "cursor/canvas";

type EerInfo = { eer: number; threshold: number; fpr: number; fnr: number; tpr: number };

type LabelBlock = {
  key: string;
  auc: number;
  roc: [number, number][];
  eer: EerInfo;
  cm_at_05: [[number, number], [number, number]];
  cm_at_eer: [[number, number], [number, number]];
  support: { n0: number; n1: number };
};

type SplitBlock = { split: string; path: string; n: number; labels: LabelBlock[] };

type CanvasData = {
  model: string;
  dataset_title: string;
  cm_threshold: number;
  splits: SplitBlock[];
};

const DATA: CanvasData = """

TSX_TAIL = r""";

const LABEL_TITLE: Record<string, string> = {
  D1: "AFB (D1)",
  D2: "TB PCR (D2)",
  D3: "Solid culture (D3)",
  D4: "Liquid culture (D4)",
  D6: "NTM (D6)",
};

function rocPathD(roc: [number, number][]): string {
  return roc.map(([fpr, tpr], i) => `${i === 0 ? "M" : "L"}${(fpr * 100).toFixed(2)},${(100 - tpr * 100).toFixed(2)}`).join(" ");
}

function CmCells({
  m,
  title,
  stroke,
  fillLo,
  fillHi,
  textPri,
  textSec,
}: {
  m: [[number, number], [number, number]];
  title: string;
  stroke: string;
  fillLo: string;
  fillHi: string;
  textPri: string;
  textSec: string;
}) {
  const flat = [m[0][0], m[0][1], m[1][0], m[1][1]];
  const vmax = Math.max(1, ...flat);
  const cell = (v: number, x: number, y: number, lab: string) => {
    const a = 0.12 + 0.88 * (v / vmax);
    return (
      <g key={lab}>
        <rect x={x} y={y} width={44} height={36} fill={fillHi} fillOpacity={a} stroke={stroke} strokeWidth={0.6} />
        <text x={x + 22} y={y + 22} textAnchor="middle" fill={textPri} fontSize={12} fontWeight="600">
          {String(v)}
        </text>
        <text x={x + 22} y={y + 33} textAnchor="middle" fill={textSec} fontSize={7}>
          {lab}
        </text>
      </g>
    );
  };
  return (
    <svg width={96} height={88} viewBox="0 0 96 88" style={{ display: "block" }}>
      <text x={0} y={10} fill={textPri} fontSize={9} fontWeight="600">
        {title}
      </text>
      {cell(m[0][0], 2, 18, "TN")}
      {cell(m[0][1], 50, 18, "FP")}
      {cell(m[1][0], 2, 56, "FN")}
      {cell(m[1][1], 50, 56, "TP")}
    </svg>
  );
}

function LabelPanel({
  row,
  stroke,
  accent,
  fillLo,
  fillHi,
  textPri,
  textSec,
}: {
  row: LabelBlock;
  stroke: string;
  accent: string;
  fillLo: string;
  fillHi: string;
  textPri: string;
  textSec: string;
}) {
  const d = rocPathD(row.roc);
  const eer = row.eer;
  return (
    <Stack gap={8}>
      <H3>
        {row.key} — {LABEL_TITLE[row.key] ?? row.key}
      </H3>
      <Text tone="secondary" size="small">
        Support: neg {row.support.n0}, pos {row.support.n1} — AUROC {row.auc.toFixed(4)} — EER {eer.eer.toFixed(4)} @ thr{" "}
        {eer.threshold.toFixed(4)}
      </Text>
      <Row gap={16} align="start" wrap>
        <Stack gap={4}>
          <Text size="small" style={{ color: textSec }}>
            ROC (threshold sweep; chance diagonal)
          </Text>
          <svg width={200} height={200} viewBox="0 0 100 100" style={{ display: "block" }}>
            <rect x={0} y={0} width={100} height={100} fill={fillLo} stroke={stroke} strokeWidth={0.4} />
            <line x1={0} y1={100} x2={100} y2={0} stroke={stroke} strokeWidth={0.35} strokeDasharray="3 3" opacity={0.7} />
            <path d={d} fill="none" stroke={accent} strokeWidth={1.2} strokeLinejoin="round" strokeLinecap="round" />
            <circle cx={eer.fpr * 100} cy={100 - eer.tpr * 100} r={2.2} fill={stroke} stroke={accent} strokeWidth={0.8} />
          </svg>
        </Stack>
        <Row gap={12} align="start">
          <CmCells
            m={row.cm_at_05}
            title={`CM @ ${String(DATA.cm_threshold)}`}
            stroke={stroke}
            fillLo={fillLo}
            fillHi={fillHi}
            textPri={textPri}
            textSec={textSec}
          />
          <CmCells m={row.cm_at_eer} title="CM @ EER thr" stroke={stroke} fillLo={fillLo} fillHi={fillHi} textPri={textPri} textSec={textSec} />
        </Row>
      </Row>
    </Stack>
  );
}

export default function VectorBalancedVersion1Canvas() {
  const { tokens: t } = useHostTheme();
  const stroke = t.stroke.primary;
  const accent = t.accent.primary;
  const fillLo = t.fill.tertiary;
  const fillHi = t.accent.control;
  const textPri = t.text.primary;
  const textSec = t.text.secondary;

  return (
    <Stack gap={20}>
      <H1>{DATA.dataset_title}</H1>
      <Text tone="secondary" size="small">
        Model: {DATA.model} — Confusion matrices: middle uses fixed threshold {DATA.cm_threshold}; right uses EER-derived threshold per label. ROC
        subsampled for display; AUROC computed on full ROC.
      </Text>
      <Divider />
      {DATA.splits.map((sp) => (
        <Stack key={sp.split} gap={14}>
          <H2>
            {sp.split.toUpperCase()} split — {sp.path} (n={sp.n})
          </H2>
          <Grid columns={1} gap={18}>
            {sp.labels.map((row) => (
              <Stack
                key={`${sp.split}-${row.key}`}
                gap={6}
                style={{
                  borderLeft: `3px solid ${accent}`,
                  paddingLeft: 12,
                  paddingTop: 4,
                  paddingBottom: 4,
                }}
              >
                <LabelPanel row={row} stroke={stroke} accent={accent} fillLo={fillLo} fillHi={fillHi} textPri={textPri} textSec={textSec} />
              </Stack>
            ))}
          </Grid>
          <Spacer height={8} />
          <Divider />
        </Stack>
      ))}
      <Text tone="secondary" size="small">
        Artifacts: TB Phase III folder — phase3_vector_balanced_version_1_features_*.npz, rf_phase3_vector_balanced_version_1.joblib,
        artifacts/metrics_phase3_vector_balanced_version_1.json, artifacts/vector_balanced_version_1/plots/*.png
      </Text>
    </Stack>
  );
}
"""


def main() -> None:
    raw = DATA_JSON.read_text(encoding="utf-8")
    json.loads(raw)  # validate
    CANVAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CANVAS_PATH.write_text(TSX_HEAD + raw + TSX_TAIL, encoding="utf-8")
    print(CANVAS_PATH)


if __name__ == "__main__":
    main()
