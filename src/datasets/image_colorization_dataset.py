from skimage import io
from torch.utils.data import Dataset


class ImageColorizationDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            data_path = self.dataset[index]
            image = io.imread(data_path)[:, :, :3]

            if self.transforms is not None:
                image = self.transforms(image)

            return image
