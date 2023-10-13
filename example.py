import torchvision
import torchvision.transforms as transforms
import torch
from kfold import KFold

transform = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])
# Here use CIFAR10 for example, you can change your own dataset.
dataset = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)

resnet18 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained = True)
fc_features = resnet18.fc.in_features
resnet18.fc = torch.nn.Linear(fc_features, 10)
model = resnet18

model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
loss_function = torch.nn.CrossEntropyLoss()
kfold = KFold(5, dataset, model, loss_function)
kfold.run(10)