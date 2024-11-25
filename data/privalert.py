from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from pathlib import Path
import csv

PRIVATE_CLASSES = ["private"]

class PrivAlert(Dataset):
    def __init__(self, root_dir, transform=transforms.Resize(240)):
        self.root_dir = Path(root_dir)
        self.data_file = self.root_dir / Path("Dataset_split") / "test_with_labels_2classes.csv"
        self.img_folder = self.root_dir / Path("Images") / "test"
        self.img_id_to_label = {}
        self.transform = transform
        self.positives = 0
        self.negatives = 0
        with open(self.data_file, 'r') as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                self.img_id_to_label[row[0].lower()] = row[1].lower()
                if row[1].lower() == "public":
                    self.negatives += 1
                if row[1].lower() == "private":
                    self.positives += 1
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        self.images = [file for file in self.img_folder.iterdir() if file.suffix.lower() in image_extensions]
        print(f"Number of positives in PrivAlert: {self.positives}")
        print(f"Number of negatives in PrivAlert: {self.negatives}")
        print(f"Number of Images in PrivAlert: {len(self.images)}")



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        file_name = img_path.stem

        label = self.img_id_to_label[file_name.split("_")[0]] # Example image name29200655307_e5f211086f_c.jpg
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

