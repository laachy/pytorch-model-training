from torchmetrics.functional import accuracy, confusion_matrix, mean_absolute_error
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

        out = {"loss": self.running_loss / self.n}
        out["acc"] = accuracy(preds, targets, "multiclass", num_classes=n_classes).item()
        out["mae"] = mean_absolute_error(preds, targets).item()
        out["cm"] = confusion_matrix(preds, targets, "multiclass", num_classes=n_classes).numpy()

        return out


        
    