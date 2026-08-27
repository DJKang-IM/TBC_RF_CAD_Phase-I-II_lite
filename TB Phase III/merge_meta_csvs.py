from pathlib import Path

import pandas as pd


def merge_csv(kn_path: Path, ne_path: Path, out_path: Path) -> None:
    d1 = pd.read_csv(kn_path, encoding="utf-8-sig")
    d2 = pd.read_csv(ne_path, encoding="utf-8-sig")
    key1 = "Study No." if "Study No." in d1.columns else ("Study ID" if "Study ID" in d1.columns else d1.columns[0])
    key2 = "Study No." if "Study No." in d2.columns else ("Study ID" if "Study ID" in d2.columns else d2.columns[0])
    d1 = d1.rename(columns={key1: "Study No."})
    d2 = d2.rename(columns={key2: "Study No."})

    cols = list(dict.fromkeys(list(d1.columns) + list(d2.columns)))
    d1 = d1.reindex(columns=cols)
    d2 = d2.reindex(columns=cols)
    m = pd.concat([d1, d2], ignore_index=True)

    def norm(v) -> str:
        s = str(v).strip()
        if s.endswith(".0") and s[:-2].isdigit():
            return s[:-2]
        if s.isdigit():
            return s
        return s

    m["Study No."] = m["Study No."].apply(norm)
    m = m[m["Study No."].astype(str).str.len() > 0]
    m = m.drop_duplicates(subset=["Study No."], keep="first").reset_index(drop=True)
    m.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"{out_path} rows={len(m)}")


def main() -> None:
    merge_csv(
        Path(r"D:\260427_KN_Meta_CSV_UTF-8.csv"),
        Path(r"D:\260427_NE_Meta_CSV_UTF-8.csv"),
        Path(r"D:\260427_ALL_Meta_CSV_UTF-8.csv"),
    )
    merge_csv(
        Path(r"D:\260427_KN_Meta_CSV_UTF-8 Version 2.csv"),
        Path(r"D:\260427_NE_Meta_CSV_UTF-8 Version 2.csv"),
        Path(r"D:\260427_ALL_Meta_CSV_UTF-8 Version 2.csv"),
    )


if __name__ == "__main__":
    main()
