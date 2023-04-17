import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    example of usage:
    my_dataset = MyDataset(combined_df)
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)
    """
    def __init__(self, combined_df):
        self.combined_df = combined_df
    
    def __len__(self):
        return len(self.combined_df)
    
    def __getitem__(self, idx):
        x_initial = self.combined_df.iloc[idx]['x_initial']
        v_initial = self.combined_df.iloc[idx]['v_initial']
        x_step20 = self.combined_df.iloc[idx]['x_step50']
        v_step20 = self.combined_df.iloc[idx]['v_step50']
        return torch.tensor([x_initial, v_initial], dtype=torch.float32), torch.tensor([x_step20, v_step20], dtype=torch.float32)
