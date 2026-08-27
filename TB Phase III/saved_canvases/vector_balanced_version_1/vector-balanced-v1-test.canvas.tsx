import { H1, H2, Stack, Table, Text } from "cursor/canvas";

export default function VectorBalancedV1TestDetail() {
  return (
    <Stack gap={12}>
      <H1>Vector Balanced Version 1 — Test</H1>
      <Text tone="secondary" size="small">n={465}. CM @ threshold 0.5 (TN, FP / FN, TP). EER row from metrics_multioutput.</Text>
      <H2>Per-label table</H2>
      <Table
        headers={["Key", "Name", "n0", "n1", "EER", "thr@EER", "FPR@EER", "FNR@EER", "TN", "FP", "FN", "TP"]}
        rows={[
    ["D1", "AFB(D1)", "310", "155", "0.0452", "0.3100", "0.0452", "0.0452", "309", "1", "25", "130"],
    ["D2", "TB PCR(D2)", "355", "110", "0.1095", "0.2967", "0.1099", "0.1091", "338", "17", "57", "53"],
    ["D3", "Solid culture(D3)", "385", "80", "0.1248", "0.1983", "0.1247", "0.1250", "380", "5", "57", "23"],
    ["D4", "Liquid culture(D4)", "363", "102", "0.1049", "0.3083", "0.1019", "0.1078", "356", "7", "63", "39"],
    ["D6", "NTM(D6)", "443", "22", "0.1823", "0.0833", "0.1828", "0.1818", "443", "0", "20", "2"]
        ]}
      />
    </Stack>
  );
}
