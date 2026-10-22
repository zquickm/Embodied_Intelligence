import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def build_loaders(args):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    full_train = datasets.MNIST(args.data_root, train=True, download=True, transform=tf)

    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(
        full_train, [len(full_train) - args.val_size, args.val_size], generator=g
    )
    test_ds = datasets.MNIST(args.data_root, train=False, download=True, transform=tf)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    return train_ld, val_ld, test_ld


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    model.train()
    return total_loss / total, correct / total

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running_loss, seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(x), y)

        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        seen += x.size(0)
    return running_loss / seen

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val_size", type=int, default=5000)
    p.add_argument("--data_root", default=r"D:\isaac\work\data")
    p.add_argument("--out_dir", default=r"D:\isaac\work")
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} seed={args.seed} amp={args.amp}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ld, val_ld, test_ld = build_loaders(args)

    model = SmallCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc, best_ep, ckpt = 0.0, 0, out / f"a1_mnist_seed{args.seed}.pt"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_ld, criterion, optimizer, scaler, device, args.amp)
        va_loss, va_acc = evaluate(model, val_ld, criterion, device)

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["val_acc"].append(va_acc)

        mark = ""
        if va_acc > best_acc:
            best_acc, best_ep = va_acc, epoch
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(), "val_acc": va_acc}, ckpt)
            mark = "  <- best"

        print(f"epoch {epoch}/{args.epochs} | {time.time()-t0:5.1f}s | "
              f"train {tr_loss:.4f} | val {va_loss:.4f} | val_acc {va_acc:.4f}{mark}")

    ck = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    te_loss, te_acc = evaluate(model, test_ld, criterion, device)
    print(f"best epoch {ck['epoch']} (val_acc {ck['val_acc']:.4f}) -> TEST acc {te_acc:.4f}")

    es = range(1, len(hist["train_loss"]) + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(es, hist["train_loss"], "o-", label="train loss")
    plt.plot(es, hist["val_loss"], "s-", label="val loss")
    plt.axvline(best_ep, ls="--", c="gray", label=f"best val_acc @ep{best_ep}")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig = out / f"a1_loss_seed{args.seed}.png"
    plt.savefig(fig, dpi=130)
    plt.close()

    print(f"saved: {ckpt.name}  {fig.name}")
    print(json.dumps({
        "seed": args.seed, "amp": args.amp, "epochs": args.epochs,
        "test_acc": round(te_acc, 4), "best_val_acc": round(best_acc, 4), "best_epoch": best_ep,
        "train_loss": [round(v, 4) for v in hist["train_loss"]],
        "val_loss": [round(v, 4) for v in hist["val_loss"]],
    }))


if __name__ == "__main__":
    main()