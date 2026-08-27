import { H1, H2, Stack, Table, Text } from "cursor/canvas";

export default function Phase3Version101TestDetail() {
  return (
    <Stack gap={12}>
      <H1>Phase III Version 1.01 — Test</H1>
      <Text tone="secondary" size="small">n={1139}. CM @ 0.5 (TN, FP / FN, TP). EER from metrics JSON.</Text>
      <H2>Per-label</H2>
      <Table
        headers={["Key", "Name", "n0", "n1", "EER", "thr@EER", "FPR@EER", "FNR@EER", "TN", "FP", "FN", "TP"]}
        rows={[
    ["D1", "AFB(D1)", "984", "155", "0.1363", "0.1800", "0.1372", "0.1355", "978", "6", "128", "27"],
    ["D2", "TB PCR(D2)", "967", "172", "0.1281", "0.1850", "0.1282", "0.1279", "958", "9", "153", "19"],
    ["D3", "Solid culture(D3)", "1016", "123", "0.1475", "0.1400", "0.1486", "0.1463", "1010", "6", "115", "8"],
    ["D4", "Liquid culture(D4)", "961", "178", "0.1394", "0.2233", "0.1384", "0.1404", "951", "10", "160", "18"],
    ["D6", "NTM(D6)", "1108", "31", "0.2495", "0.0350", "0.2410", "0.2581", "1108", "0", "29", "2"]
        ]}
      />
    </Stack>
  );
}
