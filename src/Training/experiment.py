import optuna

class Study:
    def __init__(self, study_name, direction="minimize", sampler=None):
        self.pruner()

        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction, 
            sampler=sampler or optuna.samplers.TPESampler(seed=42), # sampler defines how hparams are chosen
            pruner=self.pruner
        )
        
    def pruner(self):
        self.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=2,   # don’t prune until we have some finished trials
            n_warmup_steps=2,     # let each trial run a couple epochs before judging
            interval_steps=1      # check every epoch
        )
    
    def optimise(self, experiment, n_trials=100, show_progress_bar=True):
        self.study.optimize(experiment.train, n_trials=n_trials, show_progress_bar=show_progress_bar, callbacks=[experiment.save_best_model])  # run x times and choose new hparams
    

'''
WHAT OPTUNA NEEDS

A search space: defined by trial.suggest_*
A score to compare trials: returned by objective
A way to run an experiment: build model, train, evaluate
'''

import torch, gc, shutil
from Data.handler import ResultHandler
from Training.trainer import Trainer
from config import CKPT_NAME

class Experiment:
    def __init__(self, data_module, model_cls, model_path, epochs=50):
        self.dm = data_module
        self.model_cls = model_cls
        self.epochs = epochs
        self.model_path = model_path

        self.output_size = len(self.dm.classes())
        self.handler = ResultHandler(self.dm, tb=True, model_dir=model_path)
        self.trainer = Trainer(self.handler)

    def save_best_model(self, study, trial):
        if study.best_trial.number == trial.number:
            best_path = trial.user_attrs.get("best_path")
            if not best_path:
                return

            images, labels = next(iter(self.dm.val_dataloader(1)))
            self.handler.log_graph(images)

            dst = f"{self.model_path}/{CKPT_NAME}"
            shutil.copyfile(best_path, dst)

    def load_best_model(self):
        ckpt = torch.load(f"{self.model_path}/{CKPT_NAME}")
        hp = ckpt.get("hparams", {})
        hp.pop("batch_size")

        model = self.model_cls(len(self.dm.classes()), **hp)
        state = ckpt["state_dict"]
        model.load_state_dict(state, strict=True)

        return model

    def train(self, trial=None):
        try:            
            model, batch_size = self.model_cls.build_for_experiment(self.output_size, trial)    # create model

            # data retrieval and setup
            train_loader = self.dm.train_dataloader(batch_size)
            val_loader = self.dm.val_dataloader(batch_size)

            self.handler.set_experiment(model, trial)
            self.trainer.fit(model, train_loader, val_loader, self.epochs)  # train model

            return self.handler.best_value
        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM]: {e}. Skipping.")
            return float("inf")
        finally:
            if trial:
                self.handler.log_experiment(self.trainer.device)

            # cleanup
            self.handler.close_writers()
            del train_loader, val_loader, model
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
    
    def test(self):
        model = self.load_best_model()
        test_loader = self.dm.test_dataloader(batch_size=32)

        self.trainer.test(model, test_loader)


        

        


