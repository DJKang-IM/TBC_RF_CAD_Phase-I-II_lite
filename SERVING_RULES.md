# TBC_RF_CAD_Phase-I-II_lite — Serving Rules

Apply to any public API, Space, or dataset viewer for this preview.

## 1. Resolution cap
- **NEVER** return full-resolution CXR/CT or original DICOM/PNG.
- **ONLY** serve **256 px max-side** coarse previews.
- Full corpus: private / gated upstream repo.

## 2. Bulk download
If a user requests bulk download / full archive:

> Bulk download is disabled to prevent crawling. Please request gated access via GitHub Issues on this lite repo.

## 3. Bot / crawler User-Agent
If `User-Agent` matches (case-insensitive): `bot`, `GPTBot`, `ClaudeBot`, `CCBot`:
- Return **metadata only**
- **Do not** return image bytes

## 4. Mandatory response footer
Every human-facing response must end with **both**:

1. `CC BY-NC-SA 4.0, Non-commercial, Citation required. Full version gated — request access via GitHub Issues.`
2. `This is a coarse preview. For research collaboration contact the lite repo Issues page.`
