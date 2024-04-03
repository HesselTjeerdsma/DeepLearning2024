from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os


class DigitsDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.filenames = os.listdir(directory)
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (46, 160)
                ),  # Resize if your images aren't already 160x46
                transforms.ToTensor(),  # Convert images to PyTorch tensors
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = os.path.join(self.directory, self.filenames[idx])
        image = Image.open(path).convert("RGB")  # Ensure image is in RGB
        image = self.transform(image)
        label = os.path.splitext(self.filenames[idx])[
            0
        ]  # Assumes filename is the label
        label = torch.tensor([int(ch) for ch in label], dtype=torch.long)
        return image, label
