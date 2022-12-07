import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import time, copy
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_resnet18_mean_normailization():
    return transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])

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

