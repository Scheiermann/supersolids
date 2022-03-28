#!/usr/bin/env python

from pathlib import Path
from PIL import Image

left = 0
top = 246
right = 932
bottom = 730


search_prefix = ""
dir_path = Path.cwd()
print(dir_path)

pictures_path = sorted([x for x in dir_path.glob(search_prefix + "*.png") if x.is_file()])

for i, pic_path in enumerate(pictures_path):
    pic = Image.open(pic_path)
    pic_cropped = pic.crop((left, top, right, bottom))
    pic_cropped.save(pic_path)
    # pic_cropped.save(Path(dir_path, f"{i}.png"))
