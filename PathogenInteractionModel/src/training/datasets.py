from torch.utils.data import Dataset, DataLoader
import torch

# BERT dataset
class BERTDataset(Dataset):
    def __init__(self, c, device, embeddings, labels):
        self.c = c
        self.device = device
        self.embeddings, self.labels = embeddings, labels
            
    def __getitem__(self, idx):
        return self.embeddings[idx].to(self.device), self.labels[idx].to(self.device)
    
    def __len__(self):
        return self.embeddings.size(0)
    
def get_dataset_loaders(training_data, validation_data, testing_data, batch_size):
    return DataLoader(training_data, batch_size=batch_size, shuffle=True), DataLoader(validation_data, batch_size=batch_size, shuffle=True), DataLoader(testing_data, batch_size=batch_size, shuffle=False)

def get_LOO_dataset_loader(LOO_data, batch_size):
    return DataLoader(LOO_data, batch_size=batch_size, shuffle=False)