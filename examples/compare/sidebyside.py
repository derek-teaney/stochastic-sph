# join compare/two_level/X.png and compare/bond/X.png side by side → compare/side_X.png
import os, sys
from PIL import Image, ImageDraw, ImageFont
here = os.path.dirname(os.path.abspath(__file__))
names = sys.argv[1:] or sorted(f for f in os.listdir(os.path.join(here, "two_level")) if f.endswith(".png"))
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
except Exception:
    font = ImageFont.load_default()
for n in names:
    a, b = os.path.join(here, "two_level", n), os.path.join(here, "bond", n)
    if not (os.path.isfile(a) and os.path.isfile(b)):
        continue
    A, B = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    h = max(A.height, B.height); pad = 70
    out = Image.new("RGB", (A.width + B.width + 20, h + pad), "white")
    out.paste(A, (0, pad)); out.paste(B, (A.width + 20, pad))
    d = ImageDraw.Draw(out)
    d.text((A.width // 2 - 140, 12), "two-level (old)", fill=(31, 119, 180), font=font)
    d.text((A.width + 20 + B.width // 2 - 170, 12), "one-level bond (new)", fill=(214, 39, 40), font=font)
    out.save(os.path.join(here, "side_" + n))
    print("side_" + n)
