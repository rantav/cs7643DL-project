# Source:
# https://www.kaggle.com/datasets/delayedkarma/impressionist-classifier-data
# https://abhisheksingh007226.medium.com/how-to-classify-the-paintings-of-an-artist-using-convolutional-neural-network-87e16c0b3ee0

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


train_directory = 'data/by-artist/train'
valid_directory = 'data/by-artist/valid'

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

dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

# Size of train and validation data
dataset_sizes = {
    'train':len(dataset['train']),
    'valid':len(dataset['valid'])
}

batch_size = 8

dataloaders = {
    'train':data.DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True),
    'valid':data.DataLoader(dataset['valid'], batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
}
# Class names or target labels
class_names = dataset['train'].classes
num_classes = len(class_names)
print("Classes:", class_names)

# Print the train and validation data sizes
print("Training-set size:",dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'])
# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loading the pretrained models
model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Modify fully connected layers to match num_classes
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, num_classes)

print('Model Summary:-\n')
# for num, (name, param) in enumerate(model_ft.named_parameters()):
#     print(num, name, param.requires_grad)
summary(model_ft, input_size=(3, 224, 224))
# print(model_ft)
model_ft = model_ft.to(device)


# Loss function
criterion = nn.CrossEntropyLoss()
# Optimizer
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
# Learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Model training routine
print("\nTraining:-\n")

def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
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

num_epochs = 10
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=num_epochs)
# Save the model
PATH="saved-models/model_1.pth"
print("\nSaving the model...")
torch.save(model_ft, PATH)


EVAL_MODEL= 'saved-models/model_1.pth'
model = torch.load(EVAL_MODEL)
model.eval()

EVAL_DIR = 'data/by-artist/test'


# Prepare the eval data loader
eval_transform=transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        get_resnet18_mean_normailization()])

eval_dataset = datasets.ImageFolder(root=EVAL_DIR, transform=eval_transform)
eval_loader = data.DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Number of classes and dataset-size
num_classes=len(eval_dataset.classes)
dsize=len(eval_dataset)
# Class label names
class_names=['Cezanne', 'Degas', 'Gauguin', 'Hassam', 'Matisse', 'Monet', 'Pissarro', 'Renoir', 'Sargent', 'VanGogh']
# Initialize the prediction and label lists
predlist=torch.zeros(0,dtype=torch.long, device='cpu')
lbllist=torch.zeros(0,dtype=torch.long, device='cpu')
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
        predlist=torch.cat([predlist,predicted.view(-1).cpu()])
        lbllist=torch.cat([lbllist,labels.view(-1).cpu()])
# Overall accuracy
overall_accuracy=100 * correct / total
print('Accuracy of the network on the {:d} test images: {:.2f}%'.format(dsize,
    overall_accuracy))
# Confusion matrix
conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
print('Confusion Matrix')
print('-'*16)
print(conf_mat,'\n')