import { Divider, H1, H2, Stack, Table, Text } from "cursor/canvas";

export default function VectorBalancedV1Overview() {
  return (
    <Stack gap={16}>
      <H1>Vector Balanced Version 1</H1>
      <Text tone="secondary" size="small">
        Phase III multi-output RF. Confusion matrices and EER in metrics JSON use decision threshold 0.5 for CM; EER is from ROC geometry.
      </Text>
      <H2>AUROC by label</H2>
      <Table
        headers={["Label", "Name", "Val AUROC", "Test AUROC"]}
        rows={[
    ["D1", "AFB", "0.9949", "0.9922"],
    ["D2", "TB PCR", "0.9510", "0.9378"],
    ["D3", "Solid culture", "0.9313", "0.9407"],
    ["D4", "Liquid culture", "0.9527", "0.9545"],
    ["D6", "NTM", "0.9311", "0.8846"]
        ]}
      />
      <Divider />
      <H2>Related canvases</H2>
      <Text size="small">Same folder: vector-balanced-v1-val.canvas.tsx, vector-balanced-v1-test.canvas.tsx. Baseline (pre-balance): phase3-version-101-overview.canvas.tsx (Phase III Version 1.01).</Text>
      <Text tone="secondary" size="small">
        Disk: D:\\TB Phase III\\ — rf_phase3_vector_balanced_version_1.joblib, phase3_vector_balanced_version_1_features_*.npz, artifacts\\metrics_phase3_vector_balanced_version_1.json, artifacts\\vector_balanced_version_1\\plots\\*.png
      </Text>
    </Stack>
  );
}
