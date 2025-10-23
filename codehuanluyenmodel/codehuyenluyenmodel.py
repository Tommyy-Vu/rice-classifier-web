import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Đang sử dụng thiết bị:", device)
data_dir = "/kaggle/input/gaodataset2-2/gaodataset2"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
classes = dataset.classes
print("Các lớp:", classes)
print("Tổng số ảnh:", len(dataset))
train_indices, val_indices, test_indices = [], [], []
targets = np.array([label for _, label in dataset.samples])

for c in range(len(classes)):
    idx = np.where(targets == c)[0]
    n_total = len(idx)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    train_indices.extend(idx[:n_train])
    val_indices.extend(idx[n_train:n_train + n_val])
    test_indices.extend(idx[n_train + n_val:])

train_ds = Subset(dataset, train_indices)
val_ds = Subset(dataset, val_indices)
test_ds = Subset(dataset, test_indices)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")def count_classes(subset):
    counts = [0] * len(classes)
    for idx in subset.indices:
        _, label = dataset.samples[idx]
        counts[label] += 1
    return counts

train_counts = count_classes(train_ds)
val_counts = count_classes(val_ds)
test_counts = count_classes(test_ds)

x = np.arange(len(classes))
width = 0.25
plt.figure(figsize=(10,5))
plt.bar(x - width, train_counts, width, label='Train', color='#4CAF50')
plt.bar(x, val_counts, width, label='Val', color='#FFC107')
plt.bar(x + width, test_counts, width, label='Test', color='#2196F3')
plt.xticks(x, classes, rotation=45)
plt.xlabel("Loại gạo")
plt.ylabel("Số lượng ảnh")
plt.title("Phân bố dữ liệu Train / Validation / Test")
plt.legend()
plt.tight_layout()
plt.show()
batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size, num_workers=2, pin_memory=True)
model = models.mobilenet_v2(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model = model.to(device) 

print("Model đang chạy trên:", next(model.parameters()).device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 8
train_losses, val_losses, train_accs, val_accs = [], [], [], []

for epoch in range(epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)  # 💥 Bắt buộc
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct / total

    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_loss /= len(val_loader)
    val_acc = val_correct / val_total
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")
plt.figure(figsize=(10,8))
plt.subplot(2,1,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Biểu đồ hàm mất mát")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(2,1,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title("Biểu đồ độ chính xác")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix - MobileNetV2 Rice Classification")
plt.show()

print("           \n---Báo cáo phân loại---\n")
print(classification_report(y_true, y_pred, target_names=classes))

torch.save(model.state_dict(), "mobilenetv2_model.pth")
