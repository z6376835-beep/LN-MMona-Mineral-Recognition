import torch
from torchvision import transforms
from datasets.image_dataset import create_dataloaders, worker_init_fn
from training.train_loop import run_experiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = ""

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_loader, val_loader, test_loader, classes = create_dataloaders(data_dir, train_transform, test_transform, batch_size=64, worker_init_fn=worker_init_fn)

num_runs = 5
num_classes = len(classes)

for run_id in range(1, num_runs+1):
    run_experiment(run_id, device, train_loader, val_loader, test_loader, num_classes)
