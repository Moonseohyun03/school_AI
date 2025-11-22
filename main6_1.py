# cnn_movelet_final_with_visual.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
import pandas as pd

# -----------------------
# Dataset wrapper
# -----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X.astype(np.float32)  # (N, T, C)
        self.y = y.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]           # (T, C)
        if self.transform:
            x = self.transform(x)
        x = np.transpose(x, (1, 0))  # (C, T)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)

# -----------------------
# Simple 1D CNN
# -----------------------
class Simple1DCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x

# -----------------------
# Preprocessing
# -----------------------
def preprocess_dataset(dataset, window_size=128, max_samples=None):
    df = dataset["train"]
    xs = np.array(df["x"], dtype=float)
    ys = np.array(df["y"], dtype=float)
    zs = np.array(df["z"], dtype=float)
    labels = np.array(df["activity_code"])  # 문자열 라벨

    if max_samples:
        xs = xs[:max_samples]
        ys = ys[:max_samples]
        zs = zs[:max_samples]
        labels = labels[:max_samples]

    # 문자열 라벨 → 정수
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    total_len = len(xs)
    num_samples = total_len // window_size
    X, y = [], []

    for i in range(num_samples):
        start = i * window_size
        end = start + window_size
        signals = np.stack([xs[start:end], ys[start:end], zs[start:end]], axis=-1)
        label = Counter(labels[start:end]).most_common(1)[0][0]
        X.append(signals)
        y.append(label)

    return np.array(X), np.array(y), le  # LabelEncoder 반환

# -----------------------
# Training & Validation
# -----------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return running_loss/total, correct/total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return running_loss/total, correct/total

# -----------------------
# Main Training Routine
# -----------------------
def run_training(X, y, num_classes, epochs=30, batch_size=128, lr=1e-3, out_dir="runs"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    # Train/Val/Test split
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)

    train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=0)

    model = Simple1DCNN(in_channels=X.shape[2], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    writer = SummaryWriter(log_dir=out_dir)

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0

    for epoch in range(1, epochs+1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))

    # Test
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pth")))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print("Test Accuracy:", test_acc)

    # -----------------------
    # 논문용 시각화: Loss / Accuracy
    # -----------------------
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.legend(); plt.title("Loss")
    plt.subplot(1,2,2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.legend(); plt.title("Accuracy")
    plt.tight_layout()
    plt.show()

    writer.close()
    return model, test_acc, X_test, y_test, history

# -----------------------
# 논문용 추가 시각화
# -----------------------
def plot_results(model, X_test, y_test, le, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # 1. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()

    # 2. Precision / Recall / F1-score
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, average=None)
    df_metrics = pd.DataFrame({
        "Class": le.classes_,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Support": support
    })
    print("\n클래스별 성능 지표:\n", df_metrics)

    # 3. 막대 그래프
    x = np.arange(len(le.classes_))
    plt.figure(figsize=(10,5))
    plt.bar(x - 0.2, precision, width=0.2, label="Precision")
    plt.bar(x, recall, width=0.2, label="Recall")
    plt.bar(x + 0.2, f1, width=0.2, label="F1-score")
    plt.xticks(x, le.classes_, rotation=45)
    plt.ylabel("Score")
    plt.title("클래스별 Precision / Recall / F1-score")
    plt.legend()
    plt.show()

# -----------------------
# 실행
# -----------------------
if __name__ == "__main__":
    dataset = load_dataset("mnemoraorg/smartphone-and-smartwatch-activity-and-biometrics-15m6")
    X, y, le = preprocess_dataset(dataset, window_size=128, max_samples=50000)
    num_classes = len(np.unique(y))

    model, test_acc, X_test, y_test, history = run_training(
        X, y, num_classes=num_classes, epochs=10, batch_size=128
    )

    
    plot_results(model, X_test=X_test, y_test=y_test, le=le)
