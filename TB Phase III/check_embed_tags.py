import sys
from pathlib import Path

import pydicom
from pydicom.tag import Tag


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python check_embed_tags.py <dicom_path>")
        return 2
    p = Path(sys.argv[1])
    ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
    tags = {
        "D1": Tag(0x0011, 0x1101),
        "D2": Tag(0x0011, 0x1102),
        "D3": Tag(0x0011, 0x1103),
        "D4": Tag(0x0011, 0x1104),
        "D5": Tag(0x0011, 0x1105),
        "D6": Tag(0x0011, 0x1106),
    }
    ok = True
    for k, t in tags.items():
        if t not in ds:
            ok = False
            print(f"{k} missing tag={t}")
        else:
            try:
                v = int(ds[t].value)
            except Exception:
                v = ds[t].value
            print(f"{k} tag={t} value={v}")
    print("has_all_tags", ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

