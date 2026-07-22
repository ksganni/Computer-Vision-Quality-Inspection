"""Download sample images used in the Streamlit sidebar."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample_images"

SAMPLES = {
    "sample_street.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
    "sample_people.jpg": "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg",
}


def download(name: str, url: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    if len(data) < 100:
        raise RuntimeError(f"Download too small for {name}")
    path.write_bytes(data)
    return path


def main() -> None:
    print(f"Downloading samples into {OUT}")
    for name, url in SAMPLES.items():
        try:
            path = download(name, url)
            print(f"  ✓ {path.name} ({path.stat().st_size} bytes)")
        except Exception as exc:  # noqa: BLE001
            print(f"  ✗ {name}: {exc}")
    print("Done.")


if __name__ == "__main__":
    main()
