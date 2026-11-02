"""Download the University of Bonn EEG sets (Z,O,N,F,S) into training/data/<set>/.

The data is not redistributed in this repo; it is fetched from the original source.
Source page: https://www.ukbonn.de/epileptologie/arbeitsgruppen/ag-lehnertz-neurophysik/downloads/
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://www.ukbonn.de/site/assets/files"
SETS = {"z": 21874, "o": 21872, "n": 21871, "f": 21870, "s": 21875}
DEST = Path(__file__).resolve().parent / "data"


def main() -> None:
    for name, file_id in SETS.items():
        target = DEST / name
        if target.exists() and any(target.rglob("*.txt")):
            print(f"{name}: already present")
            continue
        url = f"{BASE}/{file_id}/{name}.zip"
        print(f"{name}: downloading {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "neurostream-data-fetch"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extractall(target)
        count = len([p for p in target.rglob("*.txt") if not p.name.startswith("._")])
        if count != 100:
            sys.exit(f"{name}: expected 100 recordings, found {count}")
    print("done")


if __name__ == "__main__":
    main()
