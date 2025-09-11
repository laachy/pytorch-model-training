import optuna

class Study:
    def __init__(self, direction="minimize", sampler=None):
        self.sh_pruner()

        self.study = optuna.create_study(
            direction=direction, 
            sampler=sampler or optuna.samplers.TPESampler(seed=42), # sampler defines how hparams are chosen
            pruner=self.pruner
        )
        
    def sh_pruner(self):
        self.pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=2,          # min epochs before considering prune
            reduction_factor=3,      # how aggressively to cut
            min_early_stopping_rate=0
        )
    
    def optimise(self, experiment, n_trials=100, show_progress_bar=True):
        self.study.optimize(experiment.run, n_trials=n_trials, show_progress_bar=show_progress_bar, callbacks=[experiment.save_best_model])  # run x times and choose new hparams
    
    



'''
WHAT OPTUNA NEEDS

A search space: defined by trial.suggest_*
A score to compare trials: returned by objective
A way to run an experiment: build model, train, evaluate
'''

import torch, gc, shutil
from Data.handler import ResultHandler
from Training.trainer import Trainer

class Experiment:
    def __init__(self, data_module, model_cls, model_path, epochs=50):
        self.dm = data_module
        self.model_cls = model_cls
        self.epochs = epochs
        self.model_path = model_path

        self.output_size = self.dm.output_size()

        self.handler = ResultHandler(self.output_size, tb=True)
        self.trainer = Trainer(self.handler)

    def save_best_model(self, study, trial):
        if study.best_trial.number == trial.number:
            best_path = trial.user_attrs.get("best_path")
            dst = f"{self.model_path}/best_model.pt"

            shutil.copyfile(best_path, dst)

    def run(self, trial=None, batch_size=32):
        try:
            tb_dir=f"{self.model_path}/tb/trial" 
            if trial:
                batch_size = trial.suggest_int("batch_size", 32, 256, step=32)
                tb_dir=f"{tb_dir}_{trial.number}"
                
            # data retrieval and setup
            train_loader = self.dm.train_dataloader(batch_size)
            val_loader = self.dm.val_dataloader(batch_size)

            model = self.model_cls.build_for_experiment(self.output_size, trial)    # create model
            self.handler.set_expiriment(model, tb_dir, trial)
            self.trainer.fit(model, train_loader, val_loader, self.epochs)  # train model

            if trial:
                self.handler.tb_log_hparams()

            return self.handler.best_value
        finally:
            # cleanup
            self.handler.close()
            del train_loader, val_loader, model
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


        

        


