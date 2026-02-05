import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrajectoryDataset(Dataset):
    def __init__(self, inputs, goals):
        # Конвертируем в тензоры PyTorch сразу
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.goals = torch.tensor(goals, dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.goals[idx]