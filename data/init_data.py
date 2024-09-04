from data.privbench import PrivBench
from torch.utils.data import DataLoader
from helpers import collate_fn

def init_data(name, path):
    if name.lower() == "privbench":
        return DataLoader(PrivBench(path), batch_size=1, shuffle=False, collate_fn=collate_fn)