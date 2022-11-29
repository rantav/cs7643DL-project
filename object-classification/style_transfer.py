import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import PIL
from PIL import Image
from style_modules import ContentLoss, StyleLoss, TotalVariationLoss
from style_utils import features_from_img, extract_features, rel_error, style_transfer
from image_utils import preprocess
import os
import random
import torchvision.transforms as T
transform = T.ToPILImage()
from torchvision.utils import save_image
dtype = torch.FloatTensor
# Uncomment out the following line if you're on a machine with a GPU set up for PyTorch!
# dtype = torch.cuda.FloatTensor

cnn = torchvision.models.squeezenet1_1(pretrained=True).features
cnn.type(dtype)

# Fix the weights of the pretrained network
for param in cnn.parameters():
    param.requires_grad = False

content_loss = ContentLoss()


# Test style loss
style_loss = StyleLoss()

# Test total variation loss
tv_loss = TotalVariationLoss()

walk_dir = "/home/xenia/afhq/val/cat"
style_dir = "/home/xenia/style"
wiki_dir = "/home/xenia/wikiart-partial/abstract"

for root, subdirs, files in os.walk(walk_dir):
    #random.shuffle(files)
    for i, filename in enumerate(files):
        if i>=50:
            break

        file_path = os.path.join(root, filename)
        style_path = f"{wiki_dir}/{random.choice(os.listdir(wiki_dir))}" #change dir name to whatever

        new_style_dir = f"{style_dir}/{style_path.split('/')[-1]}"
        im2 = Image.open(style_path)
        im2.resize((512, 512)).save(new_style_dir)

        params4 = {
            'name': 'renaissance_kids',
            'content_image': file_path,
            'style_image': new_style_dir,
            'image_size': 512,
            'style_size': 512,
            'content_layer': 3,
            'content_weight': 6e-2,
            'style_layers': [1, 4, 6, 7],
            'style_weights': [300000, 1000, 15, 3],
            'tv_weight': 2e-2,
            'content_loss': content_loss,
            'style_loss': style_loss,
            'tv_loss': tv_loss,
            'cnn': cnn,
            'dtype': dtype
        }


        res = style_transfer(**params4)
        res = res.squeeze()
        new_path = file_path.replace("/afhq/", "/afhq_styled/")
        save_image(res, new_path)
        print(f"finished file {i} from {len(files)}")

