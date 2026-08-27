import { H1, H2, Stack, Table, Text } from "cursor/canvas";

export default function Phase3Version101ValidationDetail() {
  return (
    <Stack gap={12}>
      <H1>Phase III Version 1.01 — Validation</H1>
      <Text tone="secondary" size="small">n={1117}. CM @ 0.5 (TN, FP / FN, TP). EER from metrics JSON.</Text>
      <H2>Per-label</H2>
      <Table
        headers={["Key", "Name", "n0", "n1", "EER", "thr@EER", "FPR@EER", "FNR@EER", "TN", "FP", "FN", "TP"]}
        rows={[
    ["D1", "AFB(D1)", "994", "123", "0.1223", "0.1917", "0.1227", "0.1220", "988", "6", "109", "14"],
    ["D2", "TB PCR(D2)", "986", "131", "0.1201", "0.2100", "0.1258", "0.1145", "977", "9", "116", "15"],
    ["D3", "Solid culture(D3)", "1015", "102", "0.1455", "0.1483", "0.1438", "0.1471", "1011", "4", "95", "7"],
    ["D4", "Liquid culture(D4)", "963", "154", "0.1283", "0.2333", "0.1267", "0.1299", "955", "8", "132", "22"],
    ["D6", "NTM(D6)", "1089", "28", "0.1908", "0.0400", "0.2029", "0.1786", "1089", "0", "27", "1"]
        ]}
      />
    </Stack>
  );
}
