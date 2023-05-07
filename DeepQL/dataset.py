import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    example of usage:
    my_dataset = MyDataset(combined_df, step_size=50)
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)
    """
    def __init__(self, combined_df, step_size=50):
        self.combined_df = combined_df
        self.step_size = step_size
    
    def __len__(self):
        return len(self.combined_df)
    
    def __getitem__(self, idx):
        x_initial = self.combined_df.iloc[idx]['x_initial']
        v_initial = self.combined_df.iloc[idx]['v_initial']
        x_final = self.combined_df.iloc[idx][f'x_step{self.step_size}']
        v_final = self.combined_df.iloc[idx][f'v_step{self.step_size}']
        return torch.tensor([x_initial, v_initial], dtype=torch.float32), torch.tensor([x_final, v_final], dtype=torch.float32)
