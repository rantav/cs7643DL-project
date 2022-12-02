from pathlib import Path
import numpy as np
import sys
sys.path.append("C:/Users/xenia/Desktop/cs7643DL-project")
from object_classification.style_transfer import style_transfer_our
import torch
from torch.autograd import Variable
import torchvision
import PIL
from PIL import Image
# from style_modules import ContentLoss, StyleLoss, TotalVariationLoss
from style_transfer import Params, image_loader, load_and_run_style_transfer, load_cnn
# from style_utils import features_from_img, extract_features, rel_error, style_transfer
# from image_utils import preprocess
import os
import random
import torchvision.transforms as T
transform = T.ToPILImage()
from torchvision.utils import save_image
dtype = torch.FloatTensor
# Uncomment out the following line if you're on a machine with a GPU set up for PyTorch!
# dtype = torch.cuda.FloatTensor


# content_loss = ContentLoss()


# # Test style loss
# style_loss = StyleLoss()

# # Test total variation loss
# tv_loss = TotalVariationLoss()

content_dir = "D:\\datasets\\animals-10\\val\\"
new_style_dir = "D:\\style"
new_content_dir = "D:\\content"

style = "Abstract_Expressionism"
wiki_dir = f"D:\\datasets\\wikiart\\wikiart\\{style}"
output_path = f"D:\\results\\{style}\\"
img_size = 256

for root, subdirs, files in os.walk(content_dir):
    #random.shuffle(files)
    for i, filename in enumerate(files):
        if i>=30:
            break

        content_path = os.path.join(root, filename)
        content_name = content_path.split('\\')[-1]
        new_content_img_dir = f"{new_content_dir}\\{content_name}"
        content_img = Image.open(content_path)
        img_size = min(content_img.size)
        content_img.resize((img_size, img_size)).save(new_content_img_dir)

        style_path = f"{wiki_dir}\\{random.choice(os.listdir(wiki_dir))}" #change dir name to whatever
        # style_path = f"{style_path}\\{random.choice(os.listdir(style_path))}" #change dir name to whatever
        style_name = style_path.split('\\')[-1]
        new_style_img_dir = f"{new_style_dir}\\{style_name}"
        style_img = Image.open(style_path)
        style_size = min(style_img.size)
        style_img.resize((style_size, style_size)).save(new_style_img_dir)

        output_file_path = content_path.replace(content_dir, output_path)
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)


        # num_steps = 300
        # content_weight = 1
        # style_weight = 50000
        # start_img = image_loader(new_content_img_dir, image_size=img_size)
        # params = Params(img_size, num_steps, style_weight, content_weight, start_img)
        # cnn_conf = load_cnn()
        # load_and_run_style_transfer(cnn_conf, new_style_img_dir, new_content_img_dir, output_file_path, config=params)
        # new_var = output_file_path.replace(".jpeg", "_hw.jpeg")
        style_transfer_our(dtype, None, new_content_img_dir, new_style_img_dir, output_file_path, img_size, style_size, org_size = content_img.size)
        # img  = Image.open(output_file_path)
        # img = img.resize(content_img.size)
        # img.save(output_file_path)

        print(f"finished file {i} from {len(files)}")

