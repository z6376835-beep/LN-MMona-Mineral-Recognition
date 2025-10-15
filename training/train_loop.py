import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from models.lnmmona import LNMMonaClassifier
from utils.helpers import worker_init_fn
from training.logger import get_logger
import numpy as np

logger = get_logger()

class EarlyStopping:
    def __init__(self, patience=15, delta=0.01, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            logger.info(f"Saved best model with validation loss: {val_loss:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def compute_metrics(model, loader, criterion, device, num_classes):
    model.eval()
    loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    total = len(loader.dataset)
    avg_loss = loss / total
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    with torch.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
    mA = np.nanmean(per_class_acc)
    wF1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    mF1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, mA, wF1, mF1

def run_experiment(run_id, device, train_loader, val_loader, test_loader, num_classes, num_epochs=2000):
    model = LNMMonaClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=7, min_lr=1e-6)
    early_stopping = EarlyStopping(patience=15, delta=0.001, path=f'best_model_run{run_id}.pth')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for images, labels in tqdm(train_loader, desc=f"Run {run_id} Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)                        
            train_correct += (preds == labels).sum().item()
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        val_loss, val_acc, val_mA, val_wF1, val_mF1 = compute_metrics(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
    model.load_state_dict(torch.load(f'best_model_run{run_id}.pth'))
    test_loss, test_acc, test_mA, test_wF1, test_mF1 = compute_metrics(model, test_loader, criterion, device, num_classes)
    return model, test_loss, test_acc, test_mA, test_wF1, test_mF1
