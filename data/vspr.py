from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
import random
import json

random.seed(42)

class VSPR(Dataset):
    def __init__(self, root_dir, transform=transforms.Resize(240)):
        self.root_dir = Path(root_dir) / "test2017"
        self.transform = transform
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        self.images = [file for file in self.root_dir.iterdir() if file.suffix in image_extensions]
        print(f"Number of images in VSPR: {len(self.images)}")
        self.qualitative_samples = Path(root_dir).parent / "vspr_samples"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        annotation_path = self.root_dir / f"{img_path.stem}.json"
        image = Image.open(img_path).convert("RGB")

        with open(annotation_path) as f:
            dict_annotation = json.load(f)

        # Extract class from filename
        # Assuming the class is part of the filename separated by an underscore
        # For example: "class1_image1.jpg"
        if "a0_safe" in dict_annotation['labels']:
            label = "public"
        else:
            label = "private"
        if "a16_race" in dict_annotation['labels'] and len(dict_annotation['labels']) < 3:
            image.save(self.qualitative_samples / f"a16_race_{img_path.name}")
        if "a6_hair_color" in dict_annotation['labels'] and len(dict_annotation['labels']) < 3:
            image.save(self.qualitative_samples / f"a6_hair_color_{img_path.name}")
        if "a73_landmark" in dict_annotation['labels']:
            image.save(self.qualitative_samples / f"a73_landmark_{img_path.name}")
        if "a55_religion" in dict_annotation['labels']:
            image.save(self.qualitative_samples / f"a55_religion_{img_path.name}")
        if "a31_passport" in dict_annotation['labels']:
            image.save(self.qualitative_samples / f"a31_passport_{img_path.name}")
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        return image, label, img_path
