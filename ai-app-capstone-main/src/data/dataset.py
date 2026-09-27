import torch
from torch.utils.data import Dataset

class DefectDataset(Dataset):
    def __init__(self, df):
        self.y = torch.tensor(df["label"].values, dtype=torch.long)
        self.x = torch.tensor(
            df.drop(columns=["label", "batch_id", "timestamp"]).values.astype("float32")
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]