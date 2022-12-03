import torch
from torchvision import datasets, transforms
import torch.utils.data as data
from sklearn.metrics import confusion_matrix
from PIL import Image

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
    print('Content accuracy  on the {:d} images: {:.2f}%'.format(dsize, overall_accuracy))
    # Confusion matrix
    conf_mat = confusion_matrix(lbllist.cpu().numpy(), predlist.cpu().numpy())
    print('Confusion Matrix')
    print('-'*16)
    print(conf_mat,'\n')
    return overall_accuracy


