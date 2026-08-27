from pathlib import Path

def fix_one(p: Path) -> bool:
    raw = p.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        text = raw.decode("utf-16")
    elif b"\x00" in raw[: min(200, len(raw))]:
        try:
            text = raw.decode("utf-16-le")
        except UnicodeDecodeError:
            return False
    else:
        return False
    p.write_text(text, encoding="utf-8", newline="\n")
    return True

def main() -> int:
    root = Path(r"d:\tbc-cad-ver-1.03")
    n = 0
    for p in root.rglob("*.py"):
        if fix_one(p):
            print("utf-8", p)
            n += 1
    for p in root.rglob("*.md"):
        if fix_one(p):
            n += 1
    print("done", n)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())