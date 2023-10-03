import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import copy
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score

class KFold():
    def __init__(self, k: int, dataset: Dataset, model, loss_function,
                 lr = 5e-4, batch_size = 32, num_workers = 4, drop_last = False,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.k = k
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.loss_function = loss_function
        self.device = device
        self.dataloader_list = self.generate_dataloader_list(dataset)
        self.model_list = self.generate_model_list(model)
        self.optimizer_list = self.generate_optimizer_list()

    def generate_dataloader_list(self, dataset):
            total_len = len(dataset)
            fold_len = int(total_len/self.k)
            dataloader_list = []

            for i in range(self.k):
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

                train_dataloader = DataLoader(train_set, batch_size = self.batch_size, num_workers = self.num_workers, drop_last = self.drop_last)
                valid_dataloader = DataLoader(valid_set, batch_size = self.batch_size, num_workers = self.num_workers, drop_last = self.drop_last)

                dataloader = {'train': train_dataloader, 'valid': valid_dataloader, 'valid_test': valid_dataloader}
                dataloader_list.append(dataloader)

            return dataloader_list
    
    def generate_model_list(self, model):
        model_list = []
        for i in range(self.k):
            model_list.append(copy.deepcopy(model))
        return model_list
    
    def generate_optimizer_list(self):
        optimizer_list = []
        for i in range(self.k):
            optimizer_list.append(torch.optim.AdamW(self.model_list[i].parameters(), lr = self.lr))
        return optimizer_list

    def train_epoch(self, model, dataloader, loss_function, optimizer, device):
        model.train()
        total_loss = 0
        total_accuracy = 0
        with tqdm(dataloader, unit = 'Batch', desc = 'Train') as tqdm_loader:
            for index, (image, label) in enumerate(tqdm_loader):
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
                for index, (image, label) in enumerate(tqdm_loader):
                    image = image.to(device = "cuda" if torch.cuda.is_available() else "cpu")
                    label = torch.tensor(label.to(device = device), dtype = torch.long).detach()
                
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

    def test_addition(self, image):
        with torch.no_grad():
            image = image.to(device = self.device)
            model = self.model_list[0]
            model.eval()
            predict = model(image).to(device = self.device)
            for i in range(1, self.k):
                model = self.model_list[i]
                model.eval()
                predict = predict + model(image).to(device = self.device)
            predict = predict.argmax(dim = 1)
            return predict

    def test_vote(self, image):
        with torch.no_grad():
            image = image.to(device = self.device)
            model = self.model_list[0]
            model.eval()
            predict = model(image).to(device = self.device).argmax(dim = 1)
            for i in range(1, self.k):
                model = self.model_list[i]
                model.eval()
                predict = predict + model(image).to(device = self.device).argmax(dim = 1)
            predict = predict.argmax(dim = 1)
            #if there is same votes, than return the first maximum index
            return predict

    def run(self, epoch):
        for i in range(self.k):
            print('\nFold {}'.format(i + 1))
            self.train_and_valid(self.dataloader_list[i], self.model_list[i], self.optimizer_list[i], self.loss_function, self.device, epoch)
        return self.model_list