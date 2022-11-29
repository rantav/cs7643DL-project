# from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

from __future__ import print_function
import argparse
from collections import namedtuple

import os
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image

from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# desired size of the output image
DEFAULT_IMAGE_SIZE = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

CnnConfig = namedtuple('CnnConfig', ['model', 'normalization_mean', 'normalization_std'])

def image_loader(image_name: str, image_size: int = DEFAULT_IMAGE_SIZE):
    loader = transforms.Compose([
        transforms.Resize(image_size),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)



def imshow(tensor, title=None):
    '''Show a single tensor as an image'''
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    unloader = transforms.ToPILImage()  # reconvert into PIL image
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary. Figures are pytorch tensors
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    _, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in enumerate(figures):
        tensor = figures[title]
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)      # remove the fake batch dimension
        unloader = transforms.ToPILImage()  # reconvert into PIL image
        image = unloader(image)
        axeslist.ravel()[ind].imshow(image, cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional

# plot_figures({"style": style_img, "content": content_img}, ncols=2)
# plt.figure()
# imshow(style_img, title='Style Image')

# plt.figure()
# imshow(content_img, title='Content Image')

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
CONTENT_LAYER_DEFAULT = ['conv_4']
STYLE_LAYER_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=CONTENT_LAYER_DEFAULT,
                               style_layers=STYLE_LAYER_DEFAULT):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Running style transfer..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

StartImage = Enum('StartImage', ['content', 'style', 'random'])

def load_and_run_style_transfer(cnn_conf: CnnConfig, style_image_path: str, content_image_path: str,
                                output_path: str, config):
    style_img = image_loader(style_image_path, image_size=config.image_size)
    content_img = image_loader(content_image_path, image_size=config.image_size)
    # resize style image to content image size
    style_img = transforms.Resize(content_img.shape[-2:])(style_img)

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    if config.start_image == StartImage.content:
        input_img = content_img.clone()
    elif config.start_image == StartImage.style:
        input_img = style_img.clone()
    else:
        input_img = torch.randn(content_img.data.size(), device=device)

    output = run_style_transfer(cnn_conf.model, cnn_conf.normalization_mean, cnn_conf.normalization_std,
                                content_img, style_img, input_img, num_steps=config.num_steps,
                                style_weight=config.style_weight, content_weight=config.content_weight)
    save_image(output, output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_images_dir', type=str, default='data/by-artist-4artists-256/test')
    parser.add_argument('--images_per_artist', type=int, default=5)
    parser.add_argument('--content_images_dir', type=str, default='data/content')
    parser.add_argument('--output_dir', type=str, default='data/output/style_transfered')
    # parser.add_argument('--cnn', type=str, default='vgg19', choices=['vgg19', 'vgg16']) TODO
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument('--num_steps', type=int, default=300)
    parser.add_argument('--style_weight', type=int, default=1000000)
    parser.add_argument('--content_weight', type=int, default=1)
    parser.add_argument('--start_image', type=StartImage, default=StartImage.content,
                        choices=list(StartImage))
    config = parser.parse_args()


    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    cnn_conf = CnnConfig(cnn, cnn_normalization_mean, cnn_normalization_std)
    print('CNN Summary:\n')
    summary(cnn, input_size=(3, 224, 224))


    for artist in os.listdir(config.style_images_dir):
        if artist.startswith('.'):
            continue
        per_artist = 0
        output_dir = f"{config.output_dir}/{artist}"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for style_image_id in os.listdir(f'{config.style_images_dir}/{artist}'):
            if style_image_id.startswith('.'):
                continue

            style_image_path = f"{config.style_images_dir}/{artist}/{style_image_id}"
            for content_image_id in os.listdir(config.content_images_dir):
                if content_image_id.startswith('.'):
                    continue

                content_image_path = f"{config.content_images_dir}/{content_image_id}"
                output_image_path = f"{output_dir}/style_transfer_{content_image_id}_{style_image_id}"

                print(f'\n\n>>> Processing style image: {artist}/{style_image_id} and content image {content_image_id} ...\n\n')
                if os.path.exists(output_image_path):
                    print(f'>>> Output image already exists: {output_image_path}')
                else:
                    load_and_run_style_transfer(cnn_conf, style_image_path, content_image_path, output_image_path, config=config)

            per_artist += 1
            if per_artist >= config.images_per_artist:
                break


if __name__ == '__main__':
    main()
