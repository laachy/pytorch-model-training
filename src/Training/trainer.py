import torch

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class Trainer:
    def __init__(self, handler=None):
        self.handler = handler
        self.device = get_default_device()

    def fit(self, model, train_loader, val_loader, max_epochs):
        self.model = model.to(self.device)
        self.optimiser = model.configure_optimisers()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=max_epochs)
        
        for epoch in range(max_epochs):
            self.fit_epoch(train_loader, train=True)
            self.fit_epoch(val_loader, train=False)

            self.scheduler.step()

            # logging
            if self.handler:
                if self.handler.handle(epoch) == True:
                    return

    def fit_epoch(self, loader, train: bool):
        self.model.train(train) # eval if false

        with torch.set_grad_enabled(train): # enabled only during training
            # inputs: Tensor[batch_size, channels, height, width]
            # targets: Tensor[batch_size]
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

                if self.handler:
                    self.handler.update(preds.detach().cpu(), targets.detach().cpu(), loss.item())

                del inputs, targets, outputs, loss, preds
        
        if self.handler:
            self.handler.compute(len(loader), train)