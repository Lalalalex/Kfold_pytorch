# KFold API for pytorch

* 1. [Clone Repo](#CloneRepo)
* 2. [Example Code](#ExampleCode)
* 3. [Dataset Requierment](#DatasetRequierment)
	* a. [Training](#Training)
	* b. [Testing](#Testing)
	* c. [A Sample Dataset Code](#ASampleDatasetCode)
* 4. [Tutorial](#Tutorial)
	* a. [Overview](#Overview)
	* b. [Paramaters](#Paramaters)
##  1. <a name='CloneRepo'></a>Clone Repo
```
git clone https://github.com/Lalalalex/Kfold_pytorch.git
```
```
pip install -r requirement.txt
```

##  2. <a name='ExampleCode'></a>Example Code
```python
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
```

##  3. <a name='DatasetRequierment'></a>Dataset Requierment
###  3.1. <a name='Training'></a>Training
Dataset should output [image, label].
```python
class Dataset(Dataset):
    def __getitem__(self, index):
        return images[index], labels[index]
```
###  3.2. <a name='Testing'></a>Testing
Dataset should output image.
```python
class Dataset(Dataset):
    def __getitem__(self, index):
        return images[index]
```

###  3.3. <a name='ASampleDatasetCode'></a>A Sample Dataset Code
```python
class Dataset(Dataset):
    def __init__(self, df, is_test_model = False, transforms = None):
        self.df = df
        self.is_test_model = is_test_model
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        image = self.get_image(df['path'][index])
        if self.transforms:
            image = self.transforms(image = image)['image']
        if self.is_test_model:
            return image
        label = self.df.iloc[index]['label']
        return image, label
    
    def get_image(image_path):
        image = cv2.imread(image_path)
        image_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_RGB
```

##  4. <a name='Tutorial'></a>Tutorial
###  4.1. <a name='Overview'></a>Overview
```python
class KFold():
    def __init__(self, k: int, dataset: Dataset, model, loss_function,
                 lr = 5e-4, batch_size = 32, num_workers = 4, drop_last = False,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    def generate_dataloader_list(self, dataset):
        return dataloader_list
        # Use cross validation(kfold) to generate a dataloader list
    
    def generate_model_list(self, model):
        return model_list
        # generate k independent models
    
    def generate_optimizer_list(self):
        return optimizer_list
        # generate k independent optimizer

    def train_epoch(self, model, dataloader, loss_function, optimizer, device):
        # train one model one epoch

    def valid_epoch(self, model, dataloader, loss_function, device):
        # valid one model one epoch

    def train_and_valid(self, dataloader, model, optimizer, loss_function, device, epoch):
        # train and valid one  all epochs

    def test_addition(self, image):
        return predict
        # use addition to determine predict

    def test_vote(self, image):
        return predict
        # use voting to determine predict

    def run(self, epoch):
        return self.model_list
        # run, that is, train and valid all model all epochs

```

###  4.2. <a name='Paramaters'></a>Paramaters
def __init__(self, k: int, dataset: Dataset, model, loss_function,
                 lr = 5e-4, batch_size = 32, num_workers = 4, drop_last = False,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
- **k: int**
    - The number to do cross validation
- **dataset: Dataset**
    - Your dataset
- **model: pytorch model**
    - your model
- **loss_function**
    - your loss function
- lr: float
    - Learning rate, default = 5e-4
- batch_size: int
    - Batch size, default = 16
- num_workers: int
    - The number of workers using in dataloader, default = 4
- drop_last: Boolean
    - If drop the last data could not be a batch in dataloader, default = False
- device: torch.device
    - The device to use, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu')