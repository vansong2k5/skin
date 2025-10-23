"""Utility helpers for working with user uploaded dermatology images."""

from __future__ import annotations

import base64
import io
from typing import Tuple

from PIL import Image
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
    im = Image.open(file_storage.stream).convert("RGB")
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
