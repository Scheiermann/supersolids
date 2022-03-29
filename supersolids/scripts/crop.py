#!/usr/bin/env python

from pathlib import Path
from PIL import Image

# left = 0
# top = 219
# right = 1920
# bottom = 888

left = 132
top = 0
right = 1788
bottom = 1108



search_prefix = ""
dir_path = Path("/bigwork/dscheier/results/graphs/last_frame_x_0/")
# dir_path = Path.cwd()
print(dir_path)
output_dir = "cropped"

pictures_path = sorted([x for x in dir_path.glob(search_prefix + "*.png") if x.is_file()])
if pictures_path:
    output_path = Path(pictures_path[0].parent, output_dir)
    if not output_path.is_dir():
        output_path.mkdir(parents=True)

    for i, pic_path in enumerate(pictures_path):
        output_path = Path(pic_path.parent, output_dir, pic_path.name)
        pic = Image.open(pic_path)
        pic_cropped = pic.crop((left, top, right, bottom))
        pic_cropped.save(output_path)
        # pic_cropped.save(Path(dir_path, f"{i}.png"))
else:
    print("No pictures found!")
