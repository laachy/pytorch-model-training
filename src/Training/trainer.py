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
        
        for epoch in range(max_epochs):
            self.fit_epoch(train_loader, mode="train")
            self.fit_epoch(val_loader, mode="val")

            self.scheduler.step()

            # logging
            if self.handler:
                if self.handler.handle_train(epoch) == True:
                    return

    def fit_epoch(self, loader, mode):
        train = False
        if mode == "train":
            train = True

        self.model.train(train) # eval if false

        with torch.set_grad_enabled(train): # enabled only during training
            result = Result(mode, len(loader))
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True, dtype=torch.long)
                if train:
                    self.optimiser.zero_grad(set_to_none=True)  # ensure reset gradient

                outputs = self.model(inputs)
                loss = self.model.loss(outputs, targets)

                preds = outputs.argmax(dim=1)

                if train:
                    loss.backward()
                    self.optimiser.step()   # update the parameters (perform optimization)

                result.update_batch(preds.detach().cpu(), targets.detach().cpu(), loss.item())

                del inputs, targets, outputs, loss, preds
        
        if self.handler:
            self.handler.end_epoch(result)