import { Divider, H1, H2, Stack, Table, Text } from "cursor/canvas";

export default function Phase3Version101Overview() {
  return (
    <Stack gap={16}>
      <H1>Phase III Version 1.01</H1>
      <Text tone="secondary" size="small">
        Active vs inactive cohort (before Vector Balanced Version 1). Multi-output RF. CM @ threshold 0.5; EER from ROC (metrics_multioutput). Compare with Vector Balanced Version 1 canvases (vector-balanced-v1-*.canvas.tsx).
      </Text>
      <H2>AUROC by label (Val / Test)</H2>
      <Table
        headers={["Label", "Name", "Val AUROC", "Test AUROC"]}
        rows={[
    ["D1", "AFB", "0.9387", "0.9378"],
    ["D2", "TB PCR", "0.9354", "0.9339"],
    ["D3", "Solid culture", "0.9250", "0.9240"],
    ["D4", "Liquid culture", "0.9325", "0.9328"],
    ["D6", "NTM", "0.8975", "0.8670"]
        ]}
      />
      <Divider />
      <H2>Related canvases</H2>
      <Text size="small">Same folder: phase3-version-101-val.canvas.tsx, phase3-version-101-test.canvas.tsx</Text>
      <Text tone="secondary" size="small">
        Metrics JSON: D:\\TB Phase III\\artifacts\\metrics_phase3_version_1_01.json — Model: rf_phase3_active_vs_inactive.joblib — Features: phase3_features_val.npz, phase3_features_test.npz
      </Text>
    </Stack>
  );
}
