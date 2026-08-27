## TB Phase III - DICOM 헤더에 라벨(D1~D6) 임베딩

### 무엇을 하는 스크립트인가요?
- CSV의 `Study No.`(또는 첫 번째 컬럼)를 키로 해서, 각 Study의 **6차원 라벨 벡터(D1~D6)** 를 읽습니다.
- `10144_1.dcm`, `10144_3.dcm` 처럼 파일명에서 Study ID(앞의 숫자)를 추출해 매칭합니다.
- DICOM 헤더에 **Private Tag** 로 아래 값을 저장합니다.
  - `D1~D4` (0/1)
  - `D6(NTM)`은 **파생 값**: D1~D4 중 어떤 셀이라도 `NTM` 문자열이 있으면 `D6=1`
  - `score = sum(D1..D4)` (전염성 점수)
  - `final = "NTM" if D6==1 else "TB"` (NTM이면 무조건 Override)

### 한글이 깨질 때(중요)
- CSV가 UTF-8이 아니라 **CP949(EUC-KR)** 로 저장되어 있으면, 그대로 읽으면 한글이 `����` 처럼 깨집니다.
- 이 스크립트는 CSV를 `utf-8-sig` → `cp949` → `euc-kr` 순서로 자동 시도합니다.
- DICOM 헤더에 한글 문자열을 저장할 계획이면, `(0008,0005) SpecificCharacterSet`을 `ISO_IR 192`(UTF-8)로 맞춰야 합니다.
  - 다만 일부 뷰어는 UTF-8을 제대로 표시하지 못하므로, **Private tag에는 영어/숫자만 저장**하는 것을 권장합니다.

### 설치
Python이 설치되어 있어야 합니다.

```bash
pip install -r requirements.txt
```

### 실행 예시

```bash
py embed_tb_labels_into_dicom.py ^
  --csv "251221_KN_META_251221(csv).csv" ^
  --dicom-in "DICOM_INPUT_FOLDER" ^
  --dicom-out "DICOM_OUTPUT_FOLDER" ^
  --close-all-zero
```

### CSV 컬럼이 섞여 있어 임베딩이 안 될 때(중요)
CSV에 라벨 외 컬럼(예: 최종진단)이 섞여 있으면, 자동 추정이 빗나가서 D1~D6가 엉뚱한 컬럼으로 들어갈 수 있습니다.  
이 경우 **D1~D6로 쓸 6개 컬럼을 직접 지정**하세요.

1) 컬럼 목록(인덱스 포함) 출력:

```bash
py embed_tb_labels_into_dicom.py --csv "251221_KN_META_251221(csv).csv" --dicom-in "." --dicom-out "." --print-columns
```

2) 인덱스로 지정(0-based):

```bash
py embed_tb_labels_into_dicom.py ^
  --csv "251221_KN_META_251221(csv).csv" ^
  --dicom-in "DICOM_INPUT_FOLDER" ^
  --dicom-out "DICOM_OUTPUT_FOLDER" ^
  --label-cols "1,2,3,4" ^
  --close-all-zero
```

3) 컬럼명으로 지정(정확히 일치해야 함):

```bash
py embed_tb_labels_into_dicom.py ^
  --csv "251221_KN_META_251221(csv).csv" ^
  --dicom-in "DICOM_INPUT_FOLDER" ^
  --dicom-out "DICOM_OUTPUT_FOLDER" ^
  --label-cols "도말검사,TB PCR,배양검사(고체),배양검사(액체)" ^
  --close-all-zero
```

### DICOM에 저장되는 태그
- Private Creator: `(0011,0010) = "TB_PHASE3_LABELS"`
- 벡터 문자열: `(0011,1010)` = `"D1\D2\D3\D4\D6"` 형태 (예: `"1\0\0\1\0"`)
- `D1..D4`: `(0011,1101)` ~ `(0011,1104)` (US)
- `D6(NTM)`: `(0011,1106)` (US)
- `score=sum(D1..D4)`: `(0011,1110)` (US)
- `final`: `(0011,1111)` (CS: `"NTM"` 또는 `"TB"`)

