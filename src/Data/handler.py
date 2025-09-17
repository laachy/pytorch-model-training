import optuna, torch
from torch.utils.tensorboard import SummaryWriter
from Data.early_stopper import EarlyStopper
from Utils.image_utils import plot_confusion_matrix, image_grid
from Data.data import DataModule

def _mib(b): return b/(1024**2)
def _gib(b): return b/(1024**3)
def compute_gpu_usage():
    free, total = torch.cuda.mem_get_info()
    used = total - free
    out = {
        "alloc": _mib(torch.cuda.memory_allocated()),
        "res": _mib(torch.cuda.memory_reserved()),
        "free": _gib(free),
        "total": _gib(total),
        "used": _gib(total - free)
    }
    torch.cuda.reset_peak_memory_stats()
    return out

class ResultHandler:  
    def __init__(self, dm, tb=False, model_dir=None):
        self.model_dir = model_dir
        self.tb = tb
        self.dm = dm
        self.early_stopper = EarlyStopper(patience=5, min_delta=0)

        self.writer = {}

    def set_experiment(self, model=None, trial=None):
        self.reset_experiment()

        self.trial = trial
        self.model = model
        
        # optional (for saving and logging) create writers
        if self.model_dir:
            if self.tb:
                NAMES = ["train", "val"]
                for name in NAMES:
                    path = f"{self.model_dir}/tb/{name}/trial"
                    if trial:
                        path += f"_{trial.number}"
                    self.writer[name] = SummaryWriter(path)

    def reset_experiment(self):
        self.best_epoch = -1
        self.best_value = float("inf")
        self.early_stopper.reset()

    def end_epoch(self, result):
        match result.name:     
            case "train":
                self.r_train = result
            case "val":
                self.r_val = result
            case "test":
                self.r_test = result

    # for training / validation
    def handle_train(self, epoch):
        # compute metrics
        t = self.r_train.compute_train(len(self.dm.classes()))
        v = self.r_val.compute_train(len(self.dm.classes()))

        # log
        print(f"Epoch {epoch} | train {t["loss"]:.4f}/{t["acc"]:.2%} | val {v["loss"]:.4f}/{v["acc"]:.2%}")
        if self.writer:
            # scalars
            self.log_scalars(self.writer["train"], t, epoch)
            self.log_scalars(self.writer["val"], v, epoch)
            # conf matrix
            self.log_cm(self.writer["train"], t["cm"], epoch)
            self.log_cm(self.writer["val"], v["cm"], epoch)

        val_metric = v["loss"]
        # stopping / pruning
        if self.early_stopper.early_stop(val_metric):
            return True
        if self.trial:
            self.trial.report(val_metric, epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

        # saving
        if self.best_value > val_metric:
            self.best_value = val_metric
            self.best_epoch = epoch

            if self.model and self.model_dir and self.trial:
                best_path = f"{self.model_dir}/tb/train/trial_{self.trial.number}/epoch_{epoch}.ckpt"
                ckpt = {
                    "state_dict": self.model.state_dict(),
                    "hparams": dict(self.trial.params)
                }
                torch.save(ckpt, best_path)

                self.trial.set_user_attr("best_path", best_path)

    # for testing
    def create_test_writer(self):
        if self.model_dir:
            if self.tb:
                self.writer["test"] = SummaryWriter(f"{self.model_dir}/tb/test/test")

    def handle_test(self, model, loader):
        self.create_test_writer()
        for i, (inputs, _) in enumerate(loader):
            inputs = self.dm.denorm(inputs)
            batch_preds = self.r_test.preds[i].argmax(dim=1).tolist()
            figure = image_grid(inputs, batch_preds, self.dm.classes())
            self.writer["test"].add_figure("classification results (predictions)", figure, i)

    def close_writers(self):
        for name in self.writer:
            self.writer[name].close()
    
    def log_scalars(self, writer, results, epoch):
        writer.add_scalar("loss", results["loss"], epoch)
        writer.add_scalar("accuracy", results["acc"], epoch)
        writer.add_scalar("mean average precision", results["map"], epoch)

    def log_gpu_mem(self, writer, step):
        r = compute_gpu_usage()
        print(f"GPU peak usage | {r["used"]:.1f}GiB/{r["total"]:.1f}GiB used | alloc {r["alloc"]:.0f}MiB | reserved {r["res"]:.0f}MiB")

        writer.add_scalar("GPU peak alloc MiB", r["alloc"], step)
        writer.add_scalar("GPU peak reserved MiB", r["res"], step)
        writer.add_scalar("GPU peak used GiB", r["used"], step)
        

    def log_cm(self, writer, cm, epoch):
        figure = plot_confusion_matrix(cm , self.dm.classes())
        writer.add_figure("confusion matrix", figure, epoch)

    # can change this to take hparams from model to make less coupled with optuna
    def log_experiment(self, device):
        if device.type == "cuda":
            self.log_gpu_mem(self.writer["train"], self.trial.number)

        hp = dict(self.trial.params)
        metrics = {
            "best_val_loss": self.best_value,
            "best_epoch": self.best_epoch
        }
        self.writer["train"].add_hparams(hp, metrics)

    # new writer for this (only when best model in experiment so far)
    def log_graph(self, input_data):
        writer = SummaryWriter(f"{self.model_dir}/tb/graph/trial_{self.trial.number}")
        writer.add_graph(self.model.to("cpu"), input_data)
        writer.close()
