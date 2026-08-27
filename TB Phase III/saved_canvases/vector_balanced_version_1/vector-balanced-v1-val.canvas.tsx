import { H1, H2, Stack, Table, Text } from "cursor/canvas";

export default function VectorBalancedV1ValidationDetail() {
  return (
    <Stack gap={12}>
      <H1>Vector Balanced Version 1 — Validation</H1>
      <Text tone="secondary" size="small">n={369}. CM @ threshold 0.5 (TN, FP / FN, TP). EER row from metrics_multioutput.</Text>
      <H2>Per-label table</H2>
      <Table
        headers={["Key", "Name", "n0", "n1", "EER", "thr@EER", "FPR@EER", "FNR@EER", "TN", "FP", "FN", "TP"]}
        rows={[
    ["D1", "AFB(D1)", "246", "123", "0.0366", "0.3417", "0.0407", "0.0325", "246", "0", "20", "103"],
    ["D2", "TB PCR(D2)", "288", "81", "0.1225", "0.2783", "0.1215", "0.1235", "280", "8", "43", "38"],
    ["D3", "Solid culture(D3)", "314", "55", "0.1614", "0.1867", "0.1592", "0.1636", "312", "2", "39", "16"],
    ["D4", "Liquid culture(D4)", "295", "74", "0.1100", "0.2883", "0.1119", "0.1081", "290", "5", "44", "30"],
    ["D6", "NTM(D6)", "346", "23", "0.1418", "0.0983", "0.1532", "0.1304", "346", "0", "22", "1"]
        ]}
      />
    </Stack>
  );
}
