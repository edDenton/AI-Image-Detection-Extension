"""

@author: Edward Denton
"""
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
from train import build_model, load_params, get_transforms, init_mlflow
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import numpy as np


class FixedImageFolder(ImageFolder):
    def find_classes(self, directory):
        fixed_mapping = {"fake": 0, "real": 1}
        classes = [c for c in ["fake", "real"] if (Path(directory) / c).is_dir()]
        class_to_idx = {cls: fixed_mapping[cls] for cls in classes}
        return classes, class_to_idx


def load_model(architecture, checkpoint_path, device):
    model = build_model(
        architecture=architecture,
        pretrained=False,
        freeze_backbone=False,
        device=device,
    )

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_dataloaders(generator_path, batch_size, num_workers):
    generator_path = Path(generator_path)

    test_dataset = FixedImageFolder(
        generator_path,
        transform=get_transforms(False)
    )

    labels = [label for _, label in test_dataset.samples]
    fake_count = labels.count(0)
    real_count = labels.count(1)
    print(f"[{generator_path.name}] fake: {fake_count}, real: {real_count}")

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return test_loader


def test(model, test_loader, device, generator_name):
    all_scores = []
    all_labels = []
    softmax = nn.Softmax(dim=1)

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {generator_name}", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            predictions = softmax(outputs)
            scores = predictions[:, 0]  # Gives us the confidence/probability for an image being fake/AI-generated

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_scores), np.array(all_labels)


def compute_metrics(labels: np.array, scores: np.array, threshold: float):
    # AUC
    auc = roc_auc_score(y_true=labels, y_score=scores)

    # Takes the score/confidence/probability of an image being fake
    # If the score is high/exceeds the threshold -> (scores < threshold) returns 0 which means the image is predicted fake
    # So if you have a higher threshold, the model needs to be more confident in order for the image to be classified as fake
    # Else, the score doesn't exceed the threshold -> (scores < threshold) returns 1 which means the image is predicted real
    # Ideally the model is always right but would rather have a false negative than a false positive for an image being AI-generated
    predictions = (scores < threshold).astype(int)

    # Accuracy
    accuracy = accuracy_score(y_true=labels, y_pred=predictions)

    # Precision/Recall/F1
    precision = precision_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions)

    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # Size of sample
    sample_size = len(labels)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "sample_size": sample_size
    }


def main():
    config = load_params()
    train_config = config["train"]
    data_config = config["data"]
    base_dir = Path(__file__).resolve().parent.parent

    random.seed(train_config["seed"])
    torch.manual_seed(train_config["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    checkpoint_dir = Path(base_dir / data_config["checkpoint_path"])
    model = load_model(
        architecture=train_config["architecture"],
        checkpoint_path=checkpoint_dir / "best_model.pth",
        device=device
    )

    init_mlflow(base_dir=base_dir)

    with open("../models/mlflow_run_id.txt", "r") as f:
        run_id = f.read().strip()

    with mlflow.start_run(run_id=run_id):
        test_path = base_dir / "data" / "processed" / "test"

        thresholds_to_test = [0.5, 0.55, 0.6, 0.65, 0.7]
        all_labels_combined = []
        all_scores_combined = []

        for generator in test_path.iterdir():
            loader = get_dataloaders(
                generator_path=test_path / generator.name,
                batch_size=train_config["batch_size"],
                num_workers=train_config["num_workers"]
            )

            scores, labels = test(model, loader, device, generator.name)

            for t in thresholds_to_test:
                predictions = (scores < t).astype(int)
                accuracy = accuracy_score(y_true=labels, y_pred=predictions)

                print(f"{generator.name}_t{int(t * 100)}_ACC: {accuracy:.4f}")
                mlflow.log_metric(f"{generator.name}_t{int(t * 100)}_ACC", accuracy)

            all_labels_combined.append(labels)
            all_scores_combined.append(scores)

        all_labels_combined = np.concatenate(all_labels_combined)
        all_scores_combined = np.concatenate(all_scores_combined)

        for t in thresholds_to_test:
            metrics = compute_metrics(all_labels_combined, all_scores_combined, t)
            print(f"overall_t{int(t * 100)}_ACC: {accuracy:.4f}")
            mlflow.log_metrics({
                f"overall_t{int(t * 100)}_{k}": v
                for k, v in metrics.items()
            })


if __name__ == "__main__":
    main()
