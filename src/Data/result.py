from torchmetrics.functional import accuracy, confusion_matrix
from torchmetrics.functional.classification import multiclass_average_precision
import torch

class Result:
    def __init__(self, name, size):
        self.name = name

        self.preds = []
        self.targets = []
        self.running_loss = 0.0
        self.n = size

    # update epoch metrics at the end of the batch
    def update_batch(self, preds, targets, loss):
        self.preds.append(preds)
        self.targets.append(targets)
        self.running_loss += loss

    def compute_train(self, n_classes):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        probs = torch.softmax(preds, dim=1)
        pred_labels = probs.argmax(dim=1)

        out = {"loss": self.running_loss / self.n}
        out["acc"] = accuracy(probs, targets, "multiclass", num_classes=n_classes).item()
        out["map"] = multiclass_average_precision(probs, targets, num_classes=n_classes, average="macro").item()
        out["cm"] = confusion_matrix(pred_labels, targets, "multiclass", num_classes=n_classes).numpy()

        return out


        
    