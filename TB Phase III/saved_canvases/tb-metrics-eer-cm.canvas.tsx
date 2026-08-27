import {
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Stack,
  Stat,
  Table,
  Text,
} from 'cursor/canvas';

type BinarySet = {
  name: string;
  n: number;
  support: { n0: number; n1: number };
  confusion_matrix: { labels: number[]; matrix: number[][] }; // [[TN,FP],[FN,TP]]
  eer: { eer: number; threshold: number; fpr: number; fnr: number };
};

type BinaryMetrics = {
  rf_model: string;
  threshold: number;
  sets: BinarySet[];
};

type MultiLabelEntry = {
  key: string;
  name: string;
  support: { n0: number; n1: number };
  confusion_matrix: { labels: number[]; matrix: number[][] };
  eer: { eer: number; threshold: number; fpr: number; fnr: number };
};

type MultiSet = { name: string; n: number; per_label: MultiLabelEntry[] };

type MultiMetrics = {
  rf_model: string;
  threshold: number;
  labels: { key: string; name: string }[];
  sets: MultiSet[];
};

const phase1: BinaryMetrics = {
  rf_model: 'artifacts\\rf_phase1_huge.joblib',
  threshold: 0.5,
  sets: [
    {
      name: 'phase1_features_val.npz',
      n: 2790,
      confusion_matrix: { labels: [0, 1], matrix: [[1464, 78], [193, 1055]] },
      eer: { eer: 0.10968678289933156, threshold: 0.392, fpr: 0.10959792477302205, fnr: 0.10977564102564108 },
      support: { n0: 1542, n1: 1248 },
    },
    {
      name: 'phase1_features_test.npz',
      n: 2893,
      confusion_matrix: { labels: [0, 1], matrix: [[1528, 89], [187, 1089]] },
      eer: { eer: 0.1009505683151003, threshold: 0.408, fpr: 0.10080395794681508, fnr: 0.10109717868338552 },
      support: { n0: 1617, n1: 1276 },
    },
  ],
};

const phase2: BinaryMetrics = {
  rf_model: 'artifacts\\rf_phase2_huge.joblib',
  threshold: 0.5,
  sets: [
    {
      name: 'phase2_features_val.npz',
      n: 1248,
      confusion_matrix: { labels: [0, 1], matrix: [[809, 3], [65, 371]] },
      eer: { eer: 0.04932887422605865, threshold: 0.306, fpr: 0.050492610837438424, fnr: 0.04816513761467889 },
      support: { n0: 812, n1: 436 },
    },
    {
      name: 'phase2_features_test.npz',
      n: 1276,
      confusion_matrix: { labels: [0, 1], matrix: [[799, 6], [62, 409]] },
      eer: { eer: 0.039424509765135625, threshold: 0.342, fpr: 0.03850931677018633, fnr: 0.04033970276008492 },
      support: { n0: 805, n1: 471 },
    },
  ],
};

const phase3: MultiMetrics = {
  rf_model: 'rf_phase3_active_vs_inactive.joblib',
  threshold: 0.5,
  labels: [
    { key: 'D1', name: 'AFB(D1)' },
    { key: 'D2', name: 'TB PCR(D2)' },
    { key: 'D3', name: 'Solid culture(D3)' },
    { key: 'D4', name: 'Liquid culture(D4)' },
    { key: 'D6', name: 'NTM(D6)' },
  ],
  sets: [
    {
      name: 'phase3_features_val.npz',
      n: 1117,
      per_label: [
        { key: 'D1', name: 'AFB(D1)', support: { n0: 994, n1: 123 }, confusion_matrix: { labels: [0, 1], matrix: [[988, 6], [109, 14]] }, eer: { eer: 0.12234381901163076, threshold: 0.19166666666666668, fpr: 0.1227364185110664, fnr: 0.12195121951219512 } },
        { key: 'D2', name: 'TB PCR(D2)', support: { n0: 986, n1: 131 }, confusion_matrix: { labels: [0, 1], matrix: [[977, 9], [116, 15]] }, eer: { eer: 0.1201322329405571, threshold: 0.21, fpr: 0.1257606490872211, fnr: 0.1145038167938931 } },
        { key: 'D3', name: 'Solid culture(D3)', support: { n0: 1015, n1: 102 }, confusion_matrix: { labels: [0, 1], matrix: [[1011, 4], [95, 7]] }, eer: { eer: 0.14545059403071575, threshold: 0.14833248629573695, fpr: 0.1438423645320197, fnr: 0.1470588235294118 } },
        { key: 'D4', name: 'Liquid culture(D4)', support: { n0: 963, n1: 154 }, confusion_matrix: { labels: [0, 1], matrix: [[955, 8], [132, 22]] }, eer: { eer: 0.12827878248438998, threshold: 0.23333333333333334, fpr: 0.12668743509865005, fnr: 0.1298701298701299 } },
        { key: 'D6', name: 'NTM(D6)', support: { n0: 1089, n1: 28 }, confusion_matrix: { labels: [0, 1], matrix: [[1089, 0], [27, 1]] }, eer: { eer: 0.1907549521185885, threshold: 0.04, fpr: 0.20293847566574838, fnr: 0.1785714285714286 } },
      ],
    },
    {
      name: 'phase3_features_test.npz',
      n: 1139,
      per_label: [
        { key: 'D1', name: 'AFB(D1)', support: { n0: 984, n1: 155 }, confusion_matrix: { labels: [0, 1], matrix: [[978, 6], [128, 27]] }, eer: { eer: 0.13633949645948074, threshold: 0.18, fpr: 0.13719512195121952, fnr: 0.13548387096774195 } },
        { key: 'D2', name: 'TB PCR(D2)', support: { n0: 967, n1: 172 }, confusion_matrix: { labels: [0, 1], matrix: [[958, 9], [153, 19]] }, eer: { eer: 0.12806931050239292, threshold: 0.185, fpr: 0.1282316442605998, fnr: 0.12790697674418605 } },
        { key: 'D3', name: 'Solid culture(D3)', support: { n0: 1016, n1: 123 }, confusion_matrix: { labels: [0, 1], matrix: [[1010, 6], [115, 8]] }, eer: { eer: 0.14748175532936433, threshold: 0.14, fpr: 0.1486220472440945, fnr: 0.14634146341463417 } },
        { key: 'D4', name: 'Liquid culture(D4)', support: { n0: 961, n1: 178 }, confusion_matrix: { labels: [0, 1], matrix: [[951, 10], [160, 18]] }, eer: { eer: 0.139423470401852, threshold: 0.22333333333333333, fpr: 0.1383975026014568, fnr: 0.1404494382022472 } },
        { key: 'D6', name: 'NTM(D6)', support: { n0: 1108, n1: 31 }, confusion_matrix: { labels: [0, 1], matrix: [[1108, 0], [29, 2]] }, eer: { eer: 0.24951962268545474, threshold: 0.035, fpr: 0.24097472924187727, fnr: 0.25806451612903225 } },
      ],
    },
  ],
};

function formatPct(x: number) {
  return `${(x * 100).toFixed(2)}%`;
}

function cmTable(cm: number[][]) {
  const [[tn, fp], [fn, tp]] = cm;
  return {
    headers: [' ', 'Pred 0', 'Pred 1'],
    rows: [
      ['True 0', `${tn}`, `${fp}`],
      ['True 1', `${fn}`, `${tp}`],
    ],
  };
}

function BinaryBlock({ title, m }: { title: string; m: BinaryMetrics }) {
  return (
    <Stack gap={12}>
      <H2>{title}</H2>
      <Text tone="secondary" size="small">
        Model: {m.rf_model} · Confusion matrix threshold={m.threshold}
      </Text>
      <Grid columns={m.sets.length} gap={12}>
        {m.sets.map((s) => (
          <Card key={s.name}>
            <CardHeader>{s.name}</CardHeader>
            <CardBody>
              <Stack gap={10}>
                <Grid columns={3} gap={10}>
                  <Stat value={`${s.n}`} label="N" />
                  <Stat value={`${s.support.n1}`} label="Pos(1)" />
                  <Stat value={`${s.support.n0}`} label="Neg(0)" />
                </Grid>
                <Grid columns={2} gap={10}>
                  <Stat value={formatPct(s.eer.eer)} label="EER" />
                  <Stat value={`${s.eer.threshold.toFixed(3)}`} label="EER threshold" />
                </Grid>
                <Table {...cmTable(s.confusion_matrix.matrix)} />
                <Text tone="secondary" size="small">
                  At EER: FPR={formatPct(s.eer.fpr)} · FNR={formatPct(s.eer.fnr)}
                </Text>
              </Stack>
            </CardBody>
          </Card>
        ))}
      </Grid>
    </Stack>
  );
}

function Phase3Block() {
  const sets = phase3.sets;
  const labelRows = phase3.labels.map((l) => l.name);
  return (
    <Stack gap={12}>
      <H2>Phase III Version 1.01 (Multi-label, pre–Vector Balanced)</H2>
      <Text tone="secondary" size="small">
        Model: {phase3.rf_model} · Confusion matrix threshold={phase3.threshold} · Cohort: active vs inactive (before Vector Balanced Version 1).
      </Text>
      <Grid columns={2} gap={12}>
        {sets.map((set) => {
          const rows = labelRows.map((labelName) => {
            const it = set.per_label.find((x) => x.name === labelName)!;
            const [[tn, fp], [fn, tp]] = it.confusion_matrix.matrix;
            return [
              it.name,
              `${it.support.n1}`,
              `${it.support.n0}`,
              formatPct(it.eer.eer),
              it.eer.threshold.toFixed(3),
              `${tn}`,
              `${fp}`,
              `${fn}`,
              `${tp}`,
            ];
          });
          return (
            <Card key={set.name}>
              <CardHeader>{set.name}</CardHeader>
              <CardBody>
                <Stack gap={10}>
                  <Grid columns={3} gap={10}>
                    <Stat value={`${set.n}`} label="N" />
                    <Stat value={`${phase3.threshold}`} label="CM threshold" />
                    <Stat value={`${phase3.labels.length}`} label="Labels" />
                  </Grid>
                  <Divider />
                  <H3>Per-label EER + Confusion Matrix</H3>
                  <Table
                    headers={['Label', 'Pos(1)', 'Neg(0)', 'EER', 'EER thr', 'TN', 'FP', 'FN', 'TP']}
                    rows={rows}
                  />
                </Stack>
              </CardBody>
            </Card>
          );
        })}
      </Grid>
      <Text tone="secondary" size="small">
        Label mapping: D1=AFB, D2=TB PCR, D3=고체배양(Solid), D4=액체배양(Liquid), D6=NTM.
      </Text>
    </Stack>
  );
}

export default function TbMetricsEerCm() {
  return (
    <Stack gap={18}>
      <H1>TB Models — EER & Confusion Matrix</H1>
      <Grid columns={3} gap={12}>
        <Stat value={formatPct(phase1.sets[1].eer.eer)} label="Phase I Test EER" />
        <Stat value={formatPct(phase2.sets[1].eer.eer)} label="Phase II Test EER" />
        <Stat value={formatPct(phase3.sets[1].per_label.find((x) => x.key === 'D6')!.eer.eer)} label="Phase III v1.01 Test EER (NTM/D6)" />
      </Grid>
      <Divider />
      <BinaryBlock title="Phase I (Normal vs Inactive+Active+NTM)" m={phase1} />
      <Divider />
      <BinaryBlock title="Phase II (Inactive vs Active+NTM)" m={phase2} />
      <Divider />
      <Phase3Block />
      <Divider />
      <Text tone="secondary" size="small">
        EER computed from ROC: threshold where |FPR−FNR| is minimized, EER=(FPR+FNR)/2 at that point. Confusion matrices are at threshold=0.5.
      </Text>
    </Stack>
  );
}

