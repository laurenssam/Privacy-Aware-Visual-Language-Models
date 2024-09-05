from data.privbench import PrivBench
from data.privalert import PrivAlert
from data.bivpriv import BivPriv

from torch.utils.data import DataLoader
from helpers import collate_fn

def init_data(name, path):
    if name.lower() == "privbench":
        return DataLoader(PrivBench(path), batch_size=1, shuffle=False, collate_fn=collate_fn)
    elif name.lower() == "privalert":
        return DataLoader(PrivAlert(path), batch_size=1, shuffle=False, collate_fn=collate_fn)
    elif name.lower() == "bivpriv":
        return DataLoader(BivPriv(path), batch_size=1, shuffle=False, collate_fn=collate_fn)
    else:
        raise Exception(f"Dataset {name} is not available")