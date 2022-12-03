import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import PIL
from PIL import Image
from object_classification.style_modules import ContentLoss, StyleLoss, TotalVariationLoss
from object_classification.style_utils import deprocess, features_from_img, extract_features, rel_error, style_transfer
from object_classification.image_utils import preprocess
import os
import random
import torchvision.transforms as T
transform = T.ToPILImage()
from torchvision.utils import save_image
dtype = torch.FloatTensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Uncomment out the following line if you're on a machine with a GPU set up for PyTorch!
# dtype = torch.cuda.FloatTensor

walk_dir = "/home/xenia/afhq/val"
style_dir = "/home/xenia/style"
wiki_dir = "/home/xenia/wikiart-partial/abstract"


# def deprocess(img):
#     transform = T.Compose([
#         T.Lambda(lambda x: x[0]),
#         T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
#         T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
#         T.Lambda(rescale),
#         T.ToPILImage(),
#     ])
#     return transform(img)

def style_transfer_our(dtype, cnn, content_img_path, style_img_path, out_img_path, img_size, style_size, org_size):
    cnn = torchvision.models.squeezenet1_1(pretrained=True).to(device).features
    cnn.type(dtype)

    content_loss = ContentLoss()
    style_loss = StyleLoss()
    tv_loss = TotalVariationLoss()

    # Fix the weights of the pretrained network
    for param in cnn.parameters():
        param.requires_grad = False

    params4 = {
            'name': out_img_path,
            'content_image': content_img_path,
            'style_image': style_img_path,
            'image_size': img_size,
            'style_size': style_size,
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
    res = deprocess(res.cpu())

    # content_size = Image.open(content_img_path).size
    res.resize(org_size).save(out_img_path)
    print(f"finished file {out_img_path}")
    return res

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

        res = style_transfer_our(dtype, cnn, content_loss, style_loss, tv_loss, file_path, new_style_dir)
        res = res.squeeze()
        new_path = file_path.replace("/afhq/", "/afhq_styled/")
        res = deprocess(res.cpu())
        
        print(f"finished file {i} from {len(files)}")

