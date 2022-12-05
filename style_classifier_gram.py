# Source:
# https://www.kaggle.com/datasets/delayedkarma/impressionist-classifier-data
# https://abhisheksingh007226.medium.com/how-to-classify-the-paintings-of-an-artist-using-convolutional-neural-network-87e16c0b3ee0

import argparse
from collections import namedtuple
import torch
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time, copy
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_resnet18_mean_normailization():
    return transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        get_resnet18_mean_normailization()
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        get_resnet18_mean_normailization()
    ])
}

def get_data(config):
    dataset = {
        'train': datasets.ImageFolder(root=config.train_directory, transform=image_transforms['train']),
        'valid': datasets.ImageFolder(root=config.valid_directory, transform=image_transforms['valid'])
    }

    # Size of train and validation data
    dataset_sizes = {
        'train':len(dataset['train']),
        'valid':len(dataset['valid'])
    }

    dataloaders = {
        'train':data.DataLoader(dataset['train'], batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True),
        'valid':data.DataLoader(dataset['valid'], batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    }
    # Class names or target labels
    class_names = dataset['train'].classes
    num_classes = len(class_names)
    print("Classes:", class_names)

    # Print the train and validation data sizes
    print("Training-set size:",dataset_sizes['train'],
        "\nValidation-set size:", dataset_sizes['valid'])

    return dataloaders, dataset_sizes, class_names, num_classes

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
    def __init__(self, num_classes, style_layer=0):
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
        self.model = model

    def forward(self, x):
        return self.model(x)

# STYLE_LAYER_DEFAULT = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
def create_model(num_classes, style_layer=0):
    model = StyleClassifier(num_classes, style_layer=style_layer)
    summary(model, input_size=(3, 224, 224), batch_size=8)
    return model

# Model training routine
def train_model(model, criterion, optimizer, scheduler, num_epochs=30, dataloaders=None, dataset_sizes=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # Tensorboard summary
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    with torch.autograd.set_detect_anomaly(True):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # Record training loss and accuracy for each phase
            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()
            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
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


def classify_and_report(model_path, data_path, batch_size):
    model = torch.load(model_path, map_location=torch.device(device)).to(device)
    model.eval()

    # Prepare the eval data loader
    eval_transform=transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            get_resnet18_mean_normailization()])

    eval_dataset = datasets.ImageFolder(root=data_path, transform=eval_transform)
    eval_loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    dsize = len(eval_dataset)

    # Initialize the prediction and label lists
    predlist = torch.zeros(0, dtype=torch.long, device=device)
    lbllist = torch.zeros(0, dtype=torch.long, device=device)
    # Evaluate the model accuracy on the dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in eval_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predlist = torch.cat([predlist, predicted.view(-1)])
            lbllist = torch.cat([lbllist, labels.view(-1)])
    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print('Style accuracy  on the {:d} images: {:.2f}%'.format(dsize, overall_accuracy))
    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.cpu().numpy(), predlist.cpu().numpy())
    print('Confusion Matrix')
    print('-'*16)
    print(conf_mat,'\n')
    return overall_accuracy

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
        train(config)
    elif config.task == 'train_hyper':
        for style_layer in range(15, 0, -1):
            train(config, style_layer=style_layer)
    else:
        raise ValueError(f"Unknown task {config.task}")

if __name__ == '__main__':
    main()
