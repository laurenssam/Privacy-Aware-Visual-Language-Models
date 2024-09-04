from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path

class PrivBench(Dataset):
    def __init__(self, root_dir, transform=transforms.Resize(240)):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['public', 'passport', 'face', 'tattoo', 'debit_card', 'license_plate', 'nudity', 'private_chat', 'fingerprint']
        self.private_classes = ['passport', 'face', 'tattoo', 'debit_card', 'license_plate', 'nudity', 'private_chat', 'fingerprint']
        self.images = [file for file in self.root_dir.iterdir() if file.suffix == '.jpg']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        file_name = img_path.stem

        # Extract class from filename
        # Assuming the class is part of the filename separated by an underscore
        # For example: "class1_image1.jpg"
        if "public" in file_name.lower():
            label = f"public"
        else:
            label = file_name[:file_name.rfind("_")]
            if label not in self.classes:
                print(label, self.classes)
                raise Exception("Classname incorrect")
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

