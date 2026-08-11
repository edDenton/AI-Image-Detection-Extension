"""

@author: Edward Denton
"""
# .venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///pipeline/mlflow.db to view MLFlow UI

from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import random
import yaml
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch


def load_params(params_path="../params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def init_mlflow(base_dir):
    db_path = base_dir / "models" / "v1" / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    artifact_path = base_dir / "models" / "v1" / "mlruns"
    experiment_name = "AI-Image Detector"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=f"file:///{artifact_path}"
        )
    mlflow.set_experiment(experiment_name)


def jpeg_compress(img, quality_range=(30, 90)):
    quality = random.randint(*quality_range)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def downscale_upscale(img, scale_range=(0.3, 0.8)):
    scale = random.uniform(*scale_range)
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    return small.resize((w, h), Image.Resampling.BILINEAR)


def get_transforms(augment=False):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomApply([transforms.Lambda(jpeg_compress)], p=0.3),
            transforms.RandomApply([transforms.Lambda(downscale_upscale)], p=0.3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])


def get_dataloaders(processed_path, batch_size, num_workers):
    processed_path = Path(processed_path)

    train_dataset = ImageFolder(
        root=processed_path / "train",
        transform=get_transforms(True)
    )

    val_dataset = ImageFolder(
        root=processed_path / "val",
        transform=get_transforms(False)
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print(f"Classes: {train_dataset.classes}")
    print(f"Labels: {train_dataset.class_to_idx}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    return train_loader, val_loader


def build_model(architecture, pretrained, freeze_backbone, device):
    if architecture == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)

        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features=in_features, out_features=2)

        return model.to(device)

    else:
        raise NotImplementedError(f"Model {architecture} not implemented")


def unfreeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = True
    print("Backbone unfrozen")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=" Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=" Validation", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    acc = correct / total
    return avg_loss, acc


def main():
    base_dir = Path(__file__).resolve().parent.parent

    config = load_params()
    train_config = config["train"]
    data_config = config["data"]

    random.seed(train_config["seed"])
    torch.manual_seed(train_config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    train_loader, val_loader = get_dataloaders(
        processed_path=base_dir / data_config["processed_path"],
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"]
    )

    model = build_model(
        architecture=train_config["architecture"],
        pretrained=train_config["pretrained"],
        freeze_backbone=train_config["freeze_backbone"],
        device=device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_config["learning_rate"]
    )
    scheduler = None  # Need this to be none until the backbone has been unfrozen, and so we don't step with no scheduler

    checkpoint_dir = Path(base_dir / data_config["checkpoint_path"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_checkpoint_path = checkpoint_dir / "best_model.pth"

    init_mlflow(base_dir=base_dir)
    mlflow.enable_system_metrics_logging()

    with mlflow.start_run():
        run_id_path = base_dir / "models" / "v1" / "mlflow_run_id.txt"
        with open(run_id_path, "w") as f:
            f.write(f"{mlflow.active_run().info.run_id}")

        mlflow.log_params({
            "architecture": train_config["architecture"],
            "pretrained": train_config["pretrained"],
            "freeze_backbone": train_config["freeze_backbone"],
            "unfreeze_epoch": train_config["unfreeze_epoch"],
            "epochs": train_config["epochs"],
            "batch_size": train_config["batch_size"],
            "learning_rate": train_config["learning_rate"],
            "fine_tune_lr": train_config["fine_tune_lr"],
            "num_workers": train_config["num_workers"],
            "seed": train_config["seed"]
        })

        for epoch in range(1, train_config["epochs"] + 1):
            print(f"\nEpoch {epoch}/{train_config['epochs']}")

            if train_config["freeze_backbone"] and epoch == train_config["unfreeze_epoch"]:
                unfreeze_backbone(model)
                for param in optimizer.param_groups:
                    param['lr'] = train_config["fine_tune_lr"]

                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=2
                )
                print(f"Learning Rate Updated -> {train_config['fine_tune_lr']}")
                print(f"Scheduler has been instantiated")

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            print(f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}")
            print(f"Validation loss: {val_loss:.4f}, Validation acc: {val_acc:.4f}")

            if scheduler is not None:
                scheduler.step(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f"Checkpoint saved (val_loss: {val_loss:.4f})")
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)

        mlflow.log_artifact(str(best_checkpoint_path))
        print(f"\nTraining complete. Best Validation Loss: {best_val_loss:.4f}")
        print(f"Checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
