import torch
from Data.result import Result

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# for further memory optimisations:
# https://machinelearningmastery.com/optimizing-memory-usage-pytorch-models/ 
class Trainer:
    def __init__(self, handler=None):
        self.handler = handler
        self.device = get_default_device()
        self.autocast_dtype = torch.bfloat16
    
    def test(self, model, test_loader):
        self.model = model.to(self.device)
        self.fit_epoch(test_loader, mode="test")

        if self.handler:
            self.handler.handle_test(model, test_loader)

    def fit(self, model, train_loader, val_loader, max_epochs):
        self.model = model.to(self.device)
        self.optimiser = model.configure_optimisers()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimiser, max_lr=model.lr, total_steps=max_epochs*len(train_loader))

        for epoch in range(max_epochs):
            self.fit_epoch(train_loader, mode="train")
            self.fit_epoch(val_loader, mode="val")

            # logging
            if self.handler:
                if self.handler.handle_train(epoch) == True:
                    return

    def fit_epoch(self, loader, mode):
        train = (mode == "train")
        self.model.train(train) # eval if false

        # Use inference_mode for val to save memory/overhead
        grad_ctx = torch.enable_grad() if train else torch.inference_mode()

        with grad_ctx: # enabled only during training
            result = Result(mode, len(loader))
            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True, dtype=torch.long)
                if train:
                    self.optimiser.zero_grad(set_to_none=True)  # ensure reset gradient

                with torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype):
                    outputs = self.model(inputs)
                    loss = self.model.loss(outputs, targets)

                if train:
                    loss.backward()
                    self.optimiser.step()   # update the parameters (perform optimization)
                    self.scheduler.step()

                result.update_batch(outputs.detach().cpu(), targets.detach().cpu(), loss.item())

        if self.handler:
            self.handler.end_epoch(result)