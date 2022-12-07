# Source:
# https://www.kaggle.com/datasets/delayedkarma/impressionist-classifier-data
# https://abhisheksingh007226.medium.com/how-to-classify-the-paintings-of-an-artist-using-convolutional-neural-network-87e16c0b3ee0

import argparse
import torch
from torchvision import models
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from style import get_data
from classifier import train_model, classify_and_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class GramLayer(nn.Module):

    def forward(self, input):
        self.G = gram_matrix(input)
        return input

    def get_matrix(self):
        return self.G

class StyleClassifier(nn.Module):
    def __init__(self, num_classes, style_layer=14):
        super(StyleClassifier, self).__init__()
        self.num_classes = num_classes
        base_cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device)

        self.gram_layers = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        # model = nn.Sequential(normalization)
        model = nn.Sequential()

        i = 0  # increment every time we see a conv
        for layer in base_cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name == f'conv_{style_layer}':
                gram_layer = GramLayer()
                model.add_module("gram_layer_{}".format(i), gram_layer)
                self.gram_layers.append(gram_layer)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], GramLayer):
                break
        model = model[:(i + 1)]

        # Add a last linear layer
        model.add_module('classifier', nn.Sequential(
            nn.Flatten(),
            nn.Linear(CONV_LAYER_OUTPUTS[style_layer], 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        ))
        self.model = model.to(device)

    def forward(self, x):
        return torch.clamp(self.model(x), min=-1e6, max=1e6)

def create_model(num_classes, style_layer=0):
    model = StyleClassifier(num_classes, style_layer=style_layer)
    summary(model, input_size=(3, 224, 224), batch_size=8)
    return model

CONV_LAYER_OUTPUTS = [
    3211264,
    3211264,
    3211264,
    1605632,
    1605632,
    802816,
    802816,
    802816,
    802816,
    401408,
    401408,
    401408,
    401408,
    100352,
    100352,
    100352,
]

def train(config, style_layer=0):
    print(f"\nTraining with style_layer={style_layer}...\n")

    # Get the data
    dataloaders, dataset_sizes, class_names, num_classes = get_data(config)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    model = create_model(num_classes, style_layer=style_layer)
    # Optimizer
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # Learning rate decay
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=config.train_num_epochs,
                        dataloaders=dataloaders,
                        dataset_sizes=dataset_sizes)
    print(f"\nSaving the model to {config.model_path}")
    torch.save(model, config.model_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='classify', help='train, train_hyper or classify')
    parser.add_argument('--model_path', type=str, default='saved-models/model_4artists_256.pth', help='path to save/load the model')
    parser.add_argument('--test_directory', type=str, default='data/by-artist-4artists-256/test', help='path to the test data')
    parser.add_argument('--train_directory', type=str, default='data/by-artist-4artists-256/train', help='path to the train data')
    parser.add_argument('--valid_directory', type=str, default='data/by-artist-4artists-256/valid', help='path to the valid data')
    parser.add_argument('--train_num_epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    config = parser.parse_args()

    if config.task == 'classify':
        classify_and_report(config.model_path, config.test_directory, config.batch_size)
    elif config.task == 'train':
        train(config, style_layer=14)
    elif config.task == 'train_hyper':
        for style_layer in range(15, 0, -1):
            train(config, style_layer=style_layer)
    else:
        raise ValueError(f"Unknown task {config.task}")

if __name__ == '__main__':
    main()
