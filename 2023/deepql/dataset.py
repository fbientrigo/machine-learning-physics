import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    """
    example of usage:
    my_dataset = MyDataset(combined_df, step_size=50)
    my_dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)
    """
    def __init__(self, combined_df, step_size=50, 
        x0='x_initial', xf='x_step', v0='v_initial', vf='v_step'):
        self.combined_df = combined_df
        self.step_size = step_size
        self.x0 = x0
        self.xf = xf
        self.v0 = v0
        self.vf = vf
    
    def __len__(self):
        return len(self.combined_df)
    
    def __getitem__(self, idx):
        x_initial = self.combined_df.iloc[idx][self.x0]
        v_initial = self.combined_df.iloc[idx][self.v0]
        x_final = self.combined_df.iloc[idx][self.xf+f'{self.step_size}']
        v_final = self.combined_df.iloc[idx][self.vf+f'{self.step_size}']
        return torch.tensor([x_initial, v_initial], dtype=torch.float32), torch.tensor([x_final, v_final], dtype=torch.float32)
