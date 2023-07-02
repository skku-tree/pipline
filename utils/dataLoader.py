from torch.utils.data import Dataset, DataLoader
from PIL import Image


class CreateDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[index // 8]
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label