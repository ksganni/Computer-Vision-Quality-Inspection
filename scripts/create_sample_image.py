"""Create a simple synthetic sample image for offline testing."""

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sample_images" / "sample_fruit.jpg"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # Simple colorful image (not a real fruit photo - used for smoke tests)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (40, 40, 40)

    # Draw a red "apple-like" circle and a yellow "banana-like" oval
    cv2.circle(img, (220, 240), 90, (40, 40, 220), -1)
    cv2.ellipse(img, (430, 250), (140, 50), 25, 0, 360, (40, 220, 220), -1)
    cv2.putText(
        img,
        "Sample QC Image",
        (160, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(str(OUT), img)
    print(f"Created: {OUT}")


if __name__ == "__main__":
    main()
