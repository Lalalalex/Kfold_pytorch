import torch
from torch.utils.data import DataLoader, Dataset
import copy
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class KFold:
    def __init__(self, k: int, dataset: Dataset, model, optimizer, loss_function,
                 lr = 5e-4, batch_size = 16, num_workers = 4, drop_last = False,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.k = k
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.dataloader_list = self.generate_dataloader_list(k, dataset, batch_size, num_workers, drop_last)
        self.model_list = self.generate_model_list(k)
        self.optimizer_list = self.generate_optimizer_list(k, lr, self.model_list)

    def generate_dataloader_list(self, k, dataset, batch_size, num_workers, drop_last):
            total_len = len(dataset)
            fold_len = int(total_len/k)
            dataloader_list = []

            for i in range(k):
                train_left_left_indices = 0
                train_left_right_indices = i * fold_len
                valid_left_indices = train_left_right_indices
                valid_right_indices = valid_left_indices + fold_len
                train_right_left_indices = valid_right_indices
                train_right_right_indices = total_len
                train_left_indices = list(range(train_left_left_indices, train_left_right_indices))
                train_right_indices = list(range(train_right_left_indices, train_right_right_indices))

                train_indices = train_left_indices + train_right_indices
                valid_indices = list(range(valid_left_indices, valid_right_indices))

                train_set = torch.utils.data.dataset.Subset(dataset, train_indices)
                valid_set = torch.utils.data.dataset.Subset(dataset, valid_indices)

                train_dataloader = DataLoader(train_set, batch_size = batch_size, num_workers = num_workers, drop_last = drop_last)
                valid_dataloader = DataLoader(valid_set, batch_size = batch_size, num_workers = num_workers, drop_last = drop_last)

                dataloader = {'train': train_dataloader, 'valid': valid_dataloader, 'valid_test': valid_dataloader}
                dataloader_list.append(dataloader)

            return dataloader_list
    
    def generate_model_list(self, k):
        model_list = []
        for i in range(k):
            efficientnet = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained = True)
            model_list.append(efficientnet)
        return model_list
    
    def generate_optimizer_list(self, k, lr, model_list):
        optimizer_list = []
        for i in range(k):
            optimizer_list.append(torch.optim.AdamW(model_list[i].parameters(), lr = lr))
        return optimizer_list

    def train_epoch(self, model, dataloader, loss_function, optimizer, device):
        model.train()
        total_loss = 0
        total_accuracy = 0
        with tqdm(dataloader, unit = 'Batch', desc = 'Train') as tqdm_loader:
            for index, (image_id, image, label) in enumerate(tqdm_loader):
                image = image.to(device = device)
                label = torch.tensor(label.to(device = device), dtype = torch.long)
                
                predict = model(image).to(device = device)
                loss = loss_function(predict, label)
                predict = predict.cpu().detach().argmax(dim = 1)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                loss = loss.detach().item()
                total_loss = total_loss + loss
                accuracy = accuracy_score(predict, label.cpu())
                total_accuracy = total_accuracy + accuracy
                
                tqdm_loader.set_postfix(loss = loss, average_loss = total_loss/(index + 1), average_accuracy = total_accuracy/(index + 1))

    def valid_epoch(self, model, dataloader, loss_function, device):
        model.eval()
        total_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            with tqdm(dataloader, unit = 'Batch', desc = 'Valid') as tqdm_loader:
                for index, (image_id, image, label) in enumerate(tqdm_loader):
                    image = image.to(device = "cuda" if torch.cuda.is_available() else "cpu")
                    label = torch.tensor(label.to(device = device), dtype = torch.long)
                
                    predict = model(image).to(device = device)
                    loss = loss_function(predict, label)
                    predict = predict.cpu().detach().argmax(dim = 1)
                
                    loss = loss.detach().item()
                    total_loss = total_loss + loss
                    accuracy = accuracy_score(predict, label.cpu())
                    total_accuracy = total_accuracy + accuracy
                
                    tqdm_loader.set_postfix(loss = loss, average_loss = total_loss/(index + 1), average_accuracy = total_accuracy/(index + 1))

    def train_and_valid(self, dataloader, model, optimizer, loss_function, device, epoch):
        for epoch in range(epoch):
            print('\nEpoch {}'.format(epoch + 1))
            self.train_epoch(model, dataloader['train'], loss_function, optimizer, device)
            self.valid_epoch(model, dataloader['valid'], loss_function, device)

    def run(self, k, dataloader_list, model_list, optimizer_list, loss_function, device, epoch):
        for i in range(k):
            print('\nFold {}'.format(i + 1))
            self.train_and_valid(dataloader_list[i], model_list[i], optimizer_list[i], loss_function, device, epoch)
        return model_list