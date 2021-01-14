from skimage import io
from torch.utils.data import Dataset


class ImageColorizationDataset(Dataset):
    def __init__(self, dataset, transforms=None, save_min_max=False):
        self.dataset = dataset
        self.transforms = transforms
        self.save_min_max = save_min_max

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data_path = self.dataset[index]
        image = io.imread(data_path)

        if(image.shape[-1] != 3):
            return None

        if self.transforms is not None:
            image, min_val, max_val = self.transforms(image)

        if self.save_min_max is True:
            return image, min_val, max_val

        return image
