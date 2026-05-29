import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class DefectDataset(Dataset):
    def __init__(self, qa_path: str, transform=None):
        with open(qa_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = Image.open(sample["image"]).convert("RGB")
        img = self.transform(img)
        label_text = sample["conversations"][1]["content"]
        label = 0 if "合格" in label_text and "不" not in label_text.split("合格")[0][-1:] else 1
        return img, label

def train_resnet(train_path: str, val_path: str, output_dir: str, epochs: int = 10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = DefectDataset(train_path)
    val_ds = DefectDataset(val_path)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                out = model(imgs)
                preds.extend(out.argmax(1).cpu().tolist())
                truths.extend(labels.tolist())
        acc = accuracy_score(truths, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(truths, preds, average="weighted", zero_division=0)
        print(f"Epoch {epoch+1}: acc={acc:.4f} f1={f1:.4f}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{output_dir}/resnet50_defect.pt")
    results = {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
    with open(f"{output_dir}/resnet_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    train_resnet("data/train.json", "data/val.json", "outputs/baselines/resnet")
