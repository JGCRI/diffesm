from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class CelebAHQ(Dataset):
    def __init__(self, data_dir: str):
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
            ]
        )
        self.data = ImageFolder(data_dir, transform=transform)

    def __getitem__(self, idx):
        return self.data[idx][0]

    def __len__(self):
        return len(self.data)
