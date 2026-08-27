import certifi
import ssl
import urllib.request


URL = "http<REDACTED_PATH>"


def main() -> None:
    print("certifi:", certifi.where())
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(URL, context=ctx, timeout=20) as resp:
        print("status:", resp.status)
        # Do not download whole file; just read a few bytes
        chunk = resp.read(64)
        print("read_bytes:", len(chunk))


if __name__ == "__main__":
    main()

