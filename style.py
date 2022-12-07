from torchvision import datasets, transforms
import torch.utils.data as data

from classifier import get_resnet18_mean_normailization

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

