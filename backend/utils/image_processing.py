"""Utility helpers for working with user uploaded dermatology images."""

from __future__ import annotations

import base64
import io
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageOps, ImageStat
from werkzeug.datastructures import FileStorage


def encode_image_to_base64(
    file_storage: FileStorage,
    max_w: int = 1280,
    quality: int = 85,
    return_image: bool = False,
) -> Tuple[str, str, int] | Tuple[str, str, int, Image.Image]:
    """Read an uploaded :class:`FileStorage` object and encode it as JPEG base64.

    Parameters
    ----------
    file_storage:
        Raw upload coming from Flask's request object.
    max_w:
        Maximum width to resize the image to while keeping aspect ratio. Images
        are downscaled to keep inference light-weight.
    quality:
        JPEG quality used during encoding. ``85`` is a good trade-off between
        quality and payload size for the Gemini API.
    return_image:
        When ``True`` the processed :class:`PIL.Image.Image` object is returned
        alongside the base64 payload. This is handy when the caller also needs
        to feed the image to a local ML model.

    Returns
    -------
    tuple
        A ``(base64, mime, size_bytes)`` tuple when ``return_image`` is
        ``False`` or ``(base64, mime, size_bytes, image)`` otherwise.
    """

    file_storage.stream.seek(0)
    im = Image.open(file_storage.stream)
    # Apply EXIF orientation information and immediately drop all metadata by
    # re-encoding to a new RGB buffer. This ensures location/identifying data is
    # never persisted on disk or forwarded to downstream services.
    im = ImageOps.exif_transpose(im).convert("RGB")
    if im.width > max_w:
        ratio = max_w / float(im.width)
        im = im.resize((max_w, int(im.height * ratio)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    b64 = base64.b64encode(data).decode("ascii")
    result = (b64, "image/jpeg", len(data))
    if return_image:
        # Return a copy so further operations cannot mutate the buffer that was
        # already used to produce base64.
        return (*result, im.copy())
    return result


def assess_image_quality(
    image: Image.Image,
    *,
    min_edge: int = 320,
    min_focus: float = 18.0,
    min_brightness: float = 35.0,
    max_brightness: float = 220.0,
) -> Dict[str, object]:
    """Evaluate whether the uploaded image is suitable for analysis.

    The heuristics intentionally remain lightweight so that the check can run
    within the request cycle without pulling additional ML dependencies.
    """

    width, height = image.size
    issues = []

    if min(width, height) < min_edge:
        issues.append(
            "Ảnh quá nhỏ. Vui lòng chụp cận hơn với vùng da cần tư vấn và giữ tay chắc chắn."
        )

    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    brightness = float(stat.mean[0])
    if brightness < min_brightness:
        issues.append(
            "Ảnh khá tối. Hãy chụp ở nơi đủ ánh sáng hoặc bật đèn flash dịu."
        )
    elif brightness > max_brightness:
        issues.append(
            "Ảnh bị cháy sáng. Thử giảm ánh sáng hoặc tránh ánh nắng trực tiếp."
        )

    arr = np.asarray(grayscale, dtype=np.float32)
    if arr.size:
        gy, gx = np.gradient(arr)
        focus_measure = float(np.mean(gx ** 2 + gy ** 2))
    else:
        focus_measure = 0.0

    if focus_measure < min_focus:
        issues.append(
            "Ảnh bị mờ. Vui lòng giữ máy ảnh cố định và chụp lại ở cự ly khoảng 10-15cm."
        )

    return {
        "is_acceptable": not issues,
        "issues": issues,
        "metrics": {
            "width": width,
            "height": height,
            "brightness": brightness,
            "focus": focus_measure,
        },
    }


__all__ = ["encode_image_to_base64", "assess_image_quality"]
