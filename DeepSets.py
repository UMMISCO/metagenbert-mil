import numpy as np
import pandas as pd

import time
import math
import json
import torch
from torch import nn
import torch.nn.functional as F
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import copy
import argparse

class DeepSets(nn.Module):
    def __init__(self, phi, rho, mil_layer, device):
        super(DeepSets, self).__init__()
        self.phi = phi
        self.rho = rho
        self.mil_layer = mil_layer
        self.device = device
        if mil_layer == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.phi.last_hidden_size, self.phi.last_hidden_size // 3),
                nn.Tanh(),
                nn.Linear(self.phi.last_hidden_size // 3, 1),
            ).to(self.device)
        self.criterion = (
            nn.BCEWithLogitsLoss()
            if self.rho.output_size <= 2
            else nn.CrossEntropyLoss()
        )

    def forward(self, x):
        # compute the representation for each data point
        x = self.phi.forward(x)
        A = None
        # sum up the representations
        if self.mil_layer == "sum":
            x = torch.sum(x, dim=1, keepdim=True)
        if self.mil_layer == "max":
            x = torch.max(x, dim=1, keepdim=True)[0]
        if self.mil_layer == "mean":
            x = torch.mean(x, dim=1, keepdim=True)
        if self.mil_layer == "attention":
            A = self.attention(x)
            A = F.softmax(A, dim=1)
            x = torch.bmm(torch.transpose(A, 2, 1), x)
        # compute the output
        out = self.rho.forward(x)
        return out, A

class Phi(nn.Module):
    def __init__(self, embed_size, hidden_init=200, n_layer=1, dropout=0.2):
        super(Phi, self).__init__()
        layer_size = [embed_size, hidden_init]
        n_layer -= 1
        for i in range(n_layer):
            hidden_init = hidden_init // 2
            layer_size.append(hidden_init)
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            self.layers.append(nn.LeakyReLU())
            self.layers.append(nn.Dropout(dropout))
        self.nets = nn.Sequential(*self.layers[:-1])  # Remove the last drop out
        self.last_hidden_size = layer_size[-1]

    def forward(self, x):
        return self.nets(x)


class Rho(nn.Module):
    def __init__(
        self, phi_hidden_size, hidden_init=100, n_layer=1, dropout=0.2, output_size=1
    ):
        super(Rho, self).__init__()
        self.output_size = output_size
        layer_size = [phi_hidden_size, hidden_init]
        n_layer -= 1
        for i in range(n_layer):
            hidden_init = hidden_init // 2
            layer_size.append(hidden_init)
        self.layers = []
        for i in range(len(layer_size) - 1):
            self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            self.layers.append(nn.LeakyReLU()),
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(layer_size[-1], output_size))
        self.nets = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.nets(x)



def train(samples_dir, corresp, epochs, batch_size, splits, patience, min_delta, eval_every, Phi_hidden_init, Phi_n_layer, Phi_dropout, Rho_hidden_init, Rho_n_layer, Rho_dropout, mil_layer):
    auc_list = []
    acc_list = []
    f1_list = []
    precision_list = []
    recall_list = []

    # Load data and labels
    data = []
    labels = []
    corresp_dict = json.load(open(corresp, "r"))
    for sam in os.listdir(samples_dir) : 
        if os.path.isdir(os.path.join(samples_dir, sam)):
            d=[]
            for file in os.listdir(os.path.join(samples_dir, sam)):
                if file.endswith(".pt"):
                    d.append(torch.load(os.path.join(samples_dir, sam, file)).numpy())
                elif file.endswith(".npy"):
                    d.append(np.load(os.path.join(samples_dir, sam, file)))
            d = np.array(d)
        data.append(d)
        labels.append(corresp_dict[sam])
    data = np.array(data)
    data = data.reshape(data.shape[0], data.shape[1]*data.shape[2], data.shape[3])
    print(data.shape)
    labels = np.array(labels)
    kf = StratifiedKFold(n_splits=splits, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(data, labels)):
        print(f"\nFold {fold + 1}/{splits}")

        phi = Phi(data.shape[-1], hidden_init=512, n_layer=1, dropout=0.4)
        rho = Rho(
            phi.last_hidden_size,
            hidden_init=256,
            n_layer=1,
            dropout=0.2,
            output_size=1,
        )

        model = DeepSets(phi, rho, mil_layer, "cuda")
        model.to("cuda")

        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        # Split data
        train_data, test_data = data[train_idx], data[val_idx]
        train_labels, test_labels = labels[train_idx], labels[val_idx]

        # Normalize data using train fold only
        scaler = StandardScaler()

        train_data_2d = train_data.reshape(
            train_data.shape[0] * train_data.shape[1],
            train_data.shape[2],
        )
        train_data_2d = scaler.fit_transform(train_data_2d)
        train_data = train_data_2d.reshape(
            train_data.shape[0],
            train_data.shape[1],
            train_data.shape[2],
        )

        test_data_2d = test_data.reshape(
            test_data.shape[0] * test_data.shape[1],
            test_data.shape[2],
        )
        test_data_2d = scaler.transform(test_data_2d)
        test_data = test_data_2d.reshape(
            test_data.shape[0],
            test_data.shape[1],
            test_data.shape[2],
        )

        # Convert to tensors
        train_data = torch.tensor(train_data, dtype=torch.float32).to("cuda")
        train_labels = torch.tensor(train_labels, dtype=torch.float32).to("cuda")

        test_data = torch.tensor(test_data, dtype=torch.float32).to("cuda")
        test_labels = torch.tensor(test_labels, dtype=torch.float32).to("cuda")

        n_batches = math.ceil(len(train_data) / batch_size)

        best_auc = -np.inf
        best_state_dict = None
        best_metrics = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            start_time = time.time()

            # Shuffle once per epoch, not once per batch
            perm = torch.randperm(len(train_data), device=train_data.device)
            train_data_epoch = train_data[perm]
            train_labels_epoch = train_labels[perm]

            for i in range(n_batches):
                batch_data = train_data_epoch[i * batch_size : (i + 1) * batch_size]
                batch_labels = train_labels_epoch[i * batch_size : (i + 1) * batch_size]

                optimizer.zero_grad()

                output, _ = model.forward(batch_data)
                logits = output.view(-1)

                loss = model.criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

            # Evaluate periodically
            if (epoch + 1) % eval_every == 0:
                model.eval()

                with torch.no_grad():
                    output, _ = model.forward(test_data)
                    logits = output.view(-1)

                    test_loss = model.criterion(logits, test_labels)

                    test_pred_np = torch.sigmoid(logits).detach().cpu().numpy()
                    test_labels_np = test_labels.detach().cpu().numpy()

                    test_auc = roc_auc_score(test_labels_np, test_pred_np)
                    test_acc = accuracy_score(test_labels_np, test_pred_np > 0.5)
                    test_f1 = f1_score(test_labels_np, test_pred_np > 0.5)
                    test_precision = precision_score(
                        test_labels_np,
                        test_pred_np > 0.5,
                        zero_division=0,
                    )
                    test_recall = recall_score(
                        test_labels_np,
                        test_pred_np > 0.5,
                        zero_division=0,
                    )
                    test_conf_matrix = confusion_matrix(
                        test_labels_np,
                        test_pred_np > 0.5,
                    )

                print(
                    f"Epoch {epoch + 1:03d} | "
                    f"val_loss={test_loss.item():.4f} | "
                    f"val_auc={test_auc:.4f} | "
                    f"val_acc={test_acc:.4f}"
                )

                # Early stopping is based only on validation AUC
                if test_auc > best_auc + min_delta:
                    best_auc = test_auc
                    epochs_without_improvement = 0

                    best_state_dict = copy.deepcopy(model.state_dict())

                    best_metrics = {
                        "epoch": epoch + 1,
                        "auc": test_auc,
                        "acc": test_acc,
                        "f1": test_f1,
                        "precision": test_precision,
                        "recall": test_recall,
                        "confusion_matrix": test_conf_matrix,
                    }
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping at epoch {epoch + 1}. "
                        f"Best val AUC was {best_auc:.4f} at epoch {best_metrics['epoch']}."
                    )
                    break

        # Restore the model selected by early stopping
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Store metrics from the early-stopping-selected checkpoint
        acc_list.append(best_metrics["acc"])
        auc_list.append(best_metrics["auc"])
        f1_list.append(best_metrics["f1"])
        precision_list.append(best_metrics["precision"])
        recall_list.append(best_metrics["recall"])

    print("\nCross-validation results")

    mean_auc = np.mean(auc_list)
    mean_acc = np.mean(acc_list)

    std_dev_auc = np.std(auc_list, ddof=1)
    std_dev_acc = np.std(acc_list, ddof=1)

    std_err_auc = std_dev_auc / math.sqrt(len(auc_list))
    std_err_acc = std_dev_acc / math.sqrt(len(acc_list))

    print(
        f"mean_auc={mean_auc:.4f}, "
        f"std_err_auc={std_err_auc:.4f}, "
        f"std_dev_auc={std_dev_auc:.4f}, "
        f"mean_acc={mean_acc:.4f}, "
        f"std_err_acc={std_err_acc:.4f}"
    )

def main(samples_dir, corresp, epochs, batch_size, splits, patience, min_delta, eval_every, Phi_hidden_init, Phi_n_layer, Phi_dropout, Rho_hidden_init, Rho_n_layer, Rho_dropout, mil_layer):
    train(samples_dir, corresp, epochs, batch_size, splits, patience, min_delta, eval_every, Phi_hidden_init, Phi_n_layer, Phi_dropout, Rho_hidden_init, Rho_n_layer, Rho_dropout, mil_layer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cleaned_Embed sequences using a pretrained model.")
    # Add an argument for the directory path
    parser.add_argument("samples_dir", type=str, help="Path to the directory of the embedded data (npy file containing centroids after clustering or dir containing .pt files if subsampling)")
    parser.add_argument("corresp", type=str, help="Path to the json file containing the correspondence between the samples and the labels")
    parser.add_argument("epochs", type=int, help="Number of epochs to train the model", default=500)
    parser.add_argument("batch_size", type=int, help="Batch size for training", default=32)
    parser.add_argument("splits", type=int, help="Number of splits for cross-validation", default=10)
    parser.add_argument("patience", type=int, help="Number of evaluation checks with no AUC improvement before early stopping", default=5)
    parser.add_argument("min_delta", type=float, help="Minimum AUC improvement to reset patience", default=1e-4)
    parser.add_argument("eval_every", type=int, help="Evaluate every N epochs", default=10)
    parser.add_argument("Phi_hidden_init", type=int, help="Initial hidden size for Phi", default=512)
    parser.add_argument("Phi_n_layer", type=int, help="Number of layers for Phi", default=1)
    parser.add_argument("Phi_dropout", type=float, help="Dropout rate for Phi", default=0.4)
    parser.add_argument("Rho_hidden_init", type=int, help="Initial hidden size for Rho", default=256)
    parser.add_argument("Rho_n_layer", type=int, help="Number of layers for Rho", default=1)
    parser.add_argument("Rho_dropout", type=float, help="Dropout rate for Rho", default=0.2)
    parser.add_argument("mil_layer", type=str, help="MIL layer type: sum, max, mean, attention", default="mean")
    args = parser.parse_args()
    main(args.samples_dir, args.corresp, args.epochs, args.batch_size, args.splits, args.patience, args.min_delta, args.eval_every, args.Phi_hidden_init, args.Phi_n_layer, args.Phi_dropout, args.Rho_hidden_init, args.Rho_n_layer, args.Rho_dropout, args.mil_layer)