# Source:
# https://www.kaggle.com/datasets/delayedkarma/impressionist-classifier-data
# https://abhisheksingh007226.medium.com/how-to-classify-the-paintings-of-an-artist-using-convolutional-neural-network-87e16c0b3ee0

import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from classifier import get_data, train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(num_classes):
    # Loading the pretrained models
    model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Modify fully connected layers to match num_classes
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # print('Model Summary:-\n')
    # for num, (name, param) in enumerate(model_ft.named_parameters()):
    #     print(num, name, param.requires_grad)
    # summary(model_ft, input_size=(3, 224, 224))
    # print(model_ft)
    model_ft = model_ft.to(device)
    return model_ft

def train(config):
    print("\nTraining...\n")

    # Get the data
    dataloaders, dataset_sizes, class_names, num_classes = get_data(config)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    model = create_model(num_classes)
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

