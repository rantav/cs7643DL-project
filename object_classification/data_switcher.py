import os
import random
from PIL import Image, ImageOps, ImageEnhance
import PIL
import numpy as np
import cv2
import traceback
from PIL import ImageEnhance
import shutil

path = "D:/datasets/animals_styled_5_classes"

for root, subdirs, files in os.walk(path):
    for i, filename in enumerate(files):
        dst_class = filename.rsplit("_")[0]
        src_class = root.split("\\")[-1]
        new_path = root.replace("animals_styled_5_classes", "animals_styled_per_style")
        new_path = new_path.replace(src_class, dst_class)
        new_path = f"{new_path}/{filename}"
        src = f"{root}/{filename}"
        os.makedirs(os.path.dirname(new_path), exist_ok=True, mode=0o777)
        shutil.copyfile(src, new_path)


