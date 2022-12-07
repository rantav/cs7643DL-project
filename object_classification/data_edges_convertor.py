import os
import random
from PIL import Image, ImageOps, ImageFilter
import PIL
import numpy as np
import cv2
import traceback
from PIL import ImageEnhance
import shutil


folder = "animals_styled_5_classes_stronger_2"
new_folder = f"{folder}-edges"
path = f"D:/datasets/{folder}"

for root, subdirs, files in os.walk(path):
    for i, filename in enumerate(files):
        #dst_class = filename.rsplit("_")[0]
        #src_class = root.split("\\")[-1]
        new_path = root.replace(folder, new_folder)
        #new_path = new_path.replace(src_class, dst_class)
        new_path = f"{new_path}/{filename}"
        im = Image.open(f"{root}/{filename}")
        im = im.convert("L")
 
        # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
        im = im.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                          -1, -1, -1, -1), 1, 0))
        # src = f"{root}/{filename}"
        os.makedirs(os.path.dirname(new_path), exist_ok=True, mode=0o777)
        im.save(new_path)
        # shutil.copyfile(src, new_path)


