from torch.utils.data import Dataset
from PIL import Image

class ImageDataset1(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.images = [Image.open('assets-224/' + path) for path in image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image