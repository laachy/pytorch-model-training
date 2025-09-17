import torch
from Data.result import Result

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class Trainer:
    def __init__(self, handler=None):
        self.handler = handler
        self.device = get_default_device()
    
    def test(self, model, test_loader):
        self.model = model.to(self.device)
        self.fit_epoch(test_loader, mode="test")

        if self.handler:
            self.handler.handle_test(model, test_loader)

    def fit(self, model, train_loader, val_loader, max_epochs):
        self.model = model.to(self.device)
        self.optimiser = model.configure_optimisers()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=max_epochs)
        
        if self.handler.prof: self.handler.prof.start() # hardware metrics
        for epoch in range(max_epochs):
            self.fit_epoch(train_loader, mode="train")
            self.fit_epoch(val_loader, mode="val")

            self.scheduler.step()

            # logging
            if self.handler and self.handler.handle_train(epoch) is True:
                break
        if self.handler.prof: self.handler.prof.stop()   # hardware metrics

    def fit_epoch(self, loader, mode):
        train = False
        if mode == "train":
            train = True

        self.model.train(train) # eval if false

        with torch.profiler.record_function(mode): 
            with torch.set_grad_enabled(train): # enabled only during training
                result = Result(mode, len(loader))
                for inputs, targets in loader:
                    inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                    if train:
                        self.optimiser.zero_grad(set_to_none=True)  # ensure reset gradient

                    outputs = self.model(inputs)
                    loss = self.model.loss(outputs, targets)

                    if train:
                        loss.backward()
                        self.optimiser.step()   # update the parameters (perform optimization)

                    if self.handler.prof: self.handler.prof.step()

                    result.update_batch(outputs.detach().cpu(), targets.detach().cpu(), loss.item())
            
            if self.handler:
                self.handler.end_epoch(result)