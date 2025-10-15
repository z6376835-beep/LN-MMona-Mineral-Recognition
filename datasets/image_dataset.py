from torch.utils.data import random_split
from torchvision import datasets

class TransformWrapper:
    """Wrap a dataset with a transform for train/val/test"""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        data, label = self.dataset[index]
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.dataset)

def create_dataloaders(data_dir, train_transform, test_transform, batch_size, worker_init_fn):
    """Return train/val/test DataLoaders"""
    full_dataset = datasets.ImageFolder(data_dir)
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])
    train_set = TransformWrapper(train_set, train_transform)
    val_set = TransformWrapper(val_set, test_transform)
    test_set = TransformWrapper(test_set, test_transform)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader, full_dataset.classes
