"""把 queue/*.jpg 縮到 1600px、算 id、做縮圖，輸出到 out/ 與 manifest.json"""
import base64, hashlib, io, json, sys
from pathlib import Path
from PIL import Image

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
man = []
for p in sorted(src.glob("m*_*.jpg"), key=lambda p: (p.stem.split("_")[0], int(p.stem.split("_")[1]))):
    im = Image.open(p).convert("RGB")
    sc = min(1, 1600 / max(im.size))
    im = im.resize((round(im.width * sc), round(im.height * sc)), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=88)
    data = buf.getvalue()
    pid = hashlib.sha1(data).hexdigest()[:24]
    (out / f"{pid}.jpg").write_bytes(data)
    t = im.copy(); t.thumbnail((160, 160))
    tb = io.BytesIO(); t.save(tb, "JPEG", quality=70)
    lab = p.with_suffix(".label.json")
    man.append({"id": pid, "file": str(out / f"{pid}.jpg"), "w": im.width, "h": im.height,
                "label": json.loads(lab.read_text()) if lab.exists() else None,
                "group": p.stem.split("_")[0], "n": int(p.stem.split("_")[1]),
                "thumb": "data:image/jpeg;base64," + base64.b64encode(tb.getvalue()).decode()})
(out / "manifest.json").write_text(json.dumps(man))
print(len(man), "photos")
