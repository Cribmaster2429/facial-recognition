"""Image loading and processing utilities."""

from io import BytesIO
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from fastapi import UploadFile
from PIL import Image

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


async def load_image_from_upload(upload: UploadFile) -> np.ndarray:
    """
    Load and validate an image from a FastAPI UploadFile.

    Args:
        upload: FastAPI UploadFile object.

    Returns:
        RGB numpy array of the image.

    Raises:
        ValueError: If the file is invalid, too large, or unsupported format.
    """
    if not upload.filename:
        raise ValueError("No filename provided")

    ext = Path(upload.filename).suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {ext}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    contents = await upload.read()

    if len(contents) > MAX_IMAGE_SIZE:
        raise ValueError(
            f"Image too large: {len(contents) / 1024 / 1024:.1f}MB. "
            f"Maximum size: {MAX_IMAGE_SIZE / 1024 / 1024:.0f}MB"
        )

    if len(contents) == 0:
        raise ValueError("Empty file uploaded")

    try:
        image = Image.open(BytesIO(contents))
        image.verify()
        image = Image.open(BytesIO(contents))

        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        return np.array(image)

    except Exception as e:
        raise ValueError(f"Failed to process image: {e}")


def load_image_from_path(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from a file path.

    Args:
        path: Path to the image file.

    Returns:
        RGB numpy array of the image.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is not a valid image.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {ext}. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    try:
        image = Image.open(path)

        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGB")
        elif image.mode != "RGB":
            image = image.convert("RGB")

        return np.array(image)

    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")


def save_image(image: np.ndarray, path: Union[str, Path], quality: int = 95) -> Path:
    """
    Save a numpy array as an image file.

    Args:
        image: RGB or BGR numpy array.
        path: Output file path.
        quality: JPEG quality (1-100).

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == ".png":
        cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    else:
        cv2.imwrite(str(path), image_bgr)

    return path


def draw_face_boxes(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    names: list[str] | None = None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and names on an image.

    Args:
        image: Input image (will not be modified).
        boxes: List of (top, right, bottom, left) bounding boxes.
        names: Optional list of names to display.
        color: BGR color for boxes and text.
        thickness: Line thickness.

    Returns:
        New image with drawn boxes.
    """
    output = image.copy()

    if len(output.shape) == 3 and output.shape[2] == 3:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    for i, (top, right, bottom, left) in enumerate(boxes):
        cv2.rectangle(output, (left, top), (right, bottom), color, thickness)

        if names and i < len(names):
            name = names[i]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 1

            (text_width, text_height), _ = cv2.getTextSize(
                name, font, font_scale, font_thickness
            )

            cv2.rectangle(
                output,
                (left, bottom - text_height - 10),
                (left + text_width + 6, bottom),
                color,
                cv2.FILLED,
            )

            cv2.putText(
                output,
                name,
                (left + 3, bottom - 5),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
