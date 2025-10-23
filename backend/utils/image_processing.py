import base64, io
from PIL import Image
from werkzeug.datastructures import FileStorage

def encode_image_to_base64(file_storage: FileStorage, max_w=1280, quality=85):
    file_storage.stream.seek(0)
    im = Image.open(file_storage.stream).convert("RGB")
    if im.width > max_w:
        ratio = max_w / float(im.width)
        im = im.resize((max_w, int(im.height * ratio)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    data = buf.getvalue()
    b64 = base64.b64encode(data).decode("ascii")
    # CHỈ 3 GIÁ TRỊ
    return b64, "image/jpeg", len(data)
