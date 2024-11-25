from data.privbench import PrivBench
from data.privalert import PrivAlert
from data.bivpriv import BivPriv

from torch.utils.data import DataLoader
from torchvision.datasets import Places365
from helpers import collate_fn, get_places_classes_cleaned, collate_fn_places


def init_data(name, path, shuffle=False, batch_size=1):
    if name.lower() == "privbench" or name.lower() == "privbench_hard":
        return DataLoader(PrivBench(path), batch_size=1, shuffle=shuffle, collate_fn=collate_fn)
    elif name.lower() == "privtune":
        return DataLoader(PrivBench(path), batch_size=1, shuffle=shuffle, collate_fn=collate_fn)
    elif name.lower() == "privalert":
        return DataLoader(PrivAlert(path), batch_size=1, shuffle=shuffle, collate_fn=collate_fn)
    elif name.lower() == "bivpriv":
        return DataLoader(BivPriv(path), batch_size=1, shuffle=shuffle, collate_fn=collate_fn)
    elif name.lower() == "places":
        places365 = Places365(root="/var/scratch/lsamson/Places365", small=True, download=False)
        dataloader = DataLoader(places365, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_places)
        class_names, class2idx = get_places_classes_cleaned(places365)
        return dataloader, class_names, class2idx
    else:
        raise Exception(f"Dataset {name} is not available")