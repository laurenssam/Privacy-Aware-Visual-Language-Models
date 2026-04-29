from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
import random

random.seed(42)

PRIVATE_CLASSES = [
    "Bank Statement",
    "Bill or Receipt",
    "Business Card",
    "Condom Box",
    "Debit Card",
    "Doctors Prescription",
    "Letter with Address",
    "Local News Paper",
    "Medical Record",
    "Mortage, Investment or Retirement Report",
    "Pill Bottle",
    "Pregnancy Test",
    "Tattoo",
    "Transcripts",
]


class BivPriv(Dataset):
    def __init__(self, root_dir, transform=transforms.Resize(240)):
        self.root_dir = Path(root_dir)
        self.private_img_dir = self.root_dir / "final_release"
        self.public_img_dir = self.root_dir / "Public/Original_Images_Nonprivate"
        self.transform = transform
        image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
        self.private_images = [
            file
            for file in self.private_img_dir.iterdir()
            if file.suffix.lower() in image_extensions
        ]
        self.public_images = [
            file
            for file in self.public_img_dir.iterdir()
            if file.suffix.lower() in image_extensions
        ]
        random.shuffle(self.public_images)
        self.images = (
            self.private_images + self.public_images[: len(self.private_images)]
        )
        print(f"Number of images in BIVPRIV: {len(self.images)}")

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
            label = "public"
        else:
            label = "private"
            if not any(private_class in file_name for private_class in PRIVATE_CLASSES):
                print(file_name)
                raise Exception("Image does not contain a class name")

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        return image, label, img_path
