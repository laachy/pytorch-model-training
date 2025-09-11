from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import Accuracy
from torchmetrics import MeanAbsoluteError, ConfusionMatrix
import optuna, torch

class ResultHandler:
    def __init__(self, n_classes, tb=False):
        self.tb = tb

        self.acc_metric = Accuracy(task="multiclass", num_classes=n_classes)
        self.conf_matrix = ConfusionMatrix(task="multiclass", num_classes=n_classes)
        self.mae = MeanAbsoluteError()

        self.running_loss = 0.0
        self.best_epoch = None
        self.best_value = None

    def reset_metrics(self):
        self.acc_metric.reset()
        self.conf_matrix.reset()
        self.mae.reset()
        self.running_loss = 0
            
    def set_expiriment(self, model=None, model_dir=None, trial=None):
        self.trial = trial
        self.model = model

        # optional (for saving and logging)
        self.model = model
        self.model_dir = model_dir
        self.writer = None
        if model_dir:
            if self.tb:
                self.writer = SummaryWriter(model_dir)

    def is_better(self, value):
        if self.best_value is None:
            return True
        return value < self.best_value

    def update(self, preds, targ, loss):
        self.acc_metric.update(preds, targ)
        self.conf_matrix.update(preds, targ)
        self.mae.update(preds, targ)
        self.running_loss += loss

    def compute(self, size, train: bool):
        acc = self.acc_metric.compute().item()
        cm = self.conf_matrix.compute()
        mae_val = self.mae.compute().item()
        loss = self.running_loss / size

        self.reset_metrics()

        if train:
            self.t_acc, self.t_cm, self.t_mae, self.t_loss = acc, cm, mae_val, loss
        else:
            self.v_acc, self.v_cm, self.v_mae, self.v_loss = acc, cm, mae_val, loss

    def handle(self, epoch):
        val_metric = self.v_loss

        print(f"Epoch {epoch} | train {self.t_loss:.4f}/{self.t_acc:.2%} | val {self.v_loss:.4f}/{self.v_acc:.2%}")

        if self.is_better(val_metric):
            self.best_value = val_metric
            self.best_epoch = epoch

            if self.model and self.model_dir:
                best_path = f"{self.model_dir}/epoch_{epoch}.pt"
                torch.save(self.model.state_dict(), best_path)

            if self.trial:
                self.trial.set_user_attr("best_path", best_path) 

        if self.writer:
            self.tb_log_epoch(epoch)

        if self.trial:
            self.trial.report(val_metric, epoch)

            if self.trial.should_prune():
                raise optuna.TrialPruned()

    def tb_log_epoch(self, epoch):
        self.writer.add_scalar("loss/train", self.t_loss, epoch)
        self.writer.add_scalar("loss/val", self.v_loss, epoch)
        self.writer.add_scalar("acc/train", self.t_acc, epoch)
        self.writer.add_scalar("acc/val", self.v_acc, epoch)

    # can change this to take hparams from model to make less coupled with optuna
    def tb_log_hparams(self):
        hp = dict(self.trial.params)

        metrics = {
            "best_val_loss": self.best_value,
            "best_epoch": self.best_epoch
        }

        self.writer.add_hparams(hp, metrics)

    def close(self):
        if self.writer:
            self.writer.close()



