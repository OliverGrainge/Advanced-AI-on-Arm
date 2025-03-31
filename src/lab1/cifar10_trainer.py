import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output
from pytorch_lightning.callbacks import Callback
import numpy as np


class CIFAR10Module(pl.LightningModule):
    def __init__(
        self,
        model,  # Vision Transformer model
        learning_rate: float = 1e-4,
        input_transform = None,
        batch_size: int = 64,
        num_workers: int = 0
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Standard CIFAR-10 augmentation
        self.transform = input_transform if input_transform is not None else transforms.Compose([
            transforms.RandomCrop(32, padding=4),      # Random crop with padding
            transforms.RandomHorizontalFlip(),         # Random horizontal flips
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Track metrics history
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        
        # Add these for proper validation metric collection
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def mixup_data(self, x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).cuda()

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Apply mixup
        mixed_x, y_a, y_b, lam = self.mixup_data(x, y)
        logits = self(mixed_x)
        
        # Compute mixed loss
        loss = lam * F.cross_entropy(logits, y_a, label_smoothing=0.1) + \
               (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=0.1)
        
        self.train_loss_history.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        
        # Store batch results
        self.validation_step_outputs.append({'loss': loss, 'acc': acc})
        return {'loss': loss, 'acc': acc}

    def on_validation_epoch_end(self):
        # Calculate average loss and accuracy for the epoch
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in self.validation_step_outputs]).mean()
        
        # Append to history
        self.val_loss_history.append(avg_loss.item())
        self.val_acc_history.append(avg_acc.item())
        
        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_acc', avg_acc, prog_bar=True)
        
        # Clear validation step outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.05  # Add weight decay
        )
        
        # Warmup + cosine schedule
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.learning_rate,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=len(self.train_dataloader()),
                pct_start=0.1,  # 10% warmup
                div_factor=25,  # lr_start = max_lr/25
                final_div_factor=1000,  # lr_final = lr_start/1000
            ),
            "interval": "step",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        dataset = datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=self.transform
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        dataset = datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=self.transform
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return self.val_dataloader()

class PlotlyCallback(Callback):
    def __init__(self, plot_interval=1):
        self.plot_interval = plot_interval
    
    def ema_smooth(self, scalars, weight=0.6):
        """Exponential moving average smoothing"""
        if not scalars:  # Check if list is empty
            return []
        last = scalars[0]
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if not pl_module.train_loss_history:
            return
            
        if trainer.current_epoch % self.plot_interval == 0:
            clear_output(wait=True)
            
            # Calculate steps per epoch
            steps_per_epoch = len(pl_module.train_dataloader())
            
            # Create x-axis values
            train_steps = [(i/steps_per_epoch) for i in range(len(pl_module.train_loss_history))]
            val_steps = list(range(len(pl_module.val_loss_history)))  # These are already in epochs
            
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Create slider
            steps = []
            for step in np.arange(0, 1.01, 0.1):
                train_loss_smooth = self.ema_smooth(pl_module.train_loss_history, weight=step)
                val_loss_smooth = self.ema_smooth(pl_module.val_loss_history, weight=step)
                val_acc_smooth = self.ema_smooth(pl_module.val_acc_history, weight=step)
                
                step_traces = [
                    go.Scatter(
                        x=train_steps,
                        y=train_loss_smooth,
                        name="Training Loss",
                        line=dict(color='blue'),
                        visible=False
                    ),
                    go.Scatter(
                        x=val_steps,
                        y=val_loss_smooth,
                        name="Validation Loss",
                        line=dict(color='orange'),
                        visible=False
                    ),
                    go.Scatter(
                        x=val_steps,
                        y=val_acc_smooth,
                        name="Validation Accuracy",
                        line=dict(color='green'),
                        visible=False
                    )
                ]
                
                steps.append(dict(
                    method="update",
                    args=[{"visible": [False] * len(fig.data)}],
                    label=f"{step:.1f}"
                ))
                
                for trace in step_traces:
                    fig.add_trace(trace, secondary_y=(trace.name == "Validation Accuracy"))
            
            # Make the first set of traces visible
            for i in range(3):
                fig.data[i].visible = True
            
            # Update steps to show correct traces
            for i, step in enumerate(steps):
                step["args"][0]["visible"] = [False] * len(fig.data)
                step["args"][0]["visible"][i*3:(i*3)+3] = [True, True, True]
            
            # Add slider
            sliders = [dict(
                active=0,
                currentvalue={"prefix": "Smoothing: "},
                pad={"t": 50},
                steps=steps
            )]
            
            # Set titles and layout
            fig.update_layout(
                title="Training Progress",
                xaxis_title="Epochs",
                hovermode='x unified',
                sliders=sliders,
                showlegend=True,
                height=600
            )
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Loss", secondary_y=False)
            fig.update_yaxes(title_text="Accuracy", secondary_y=True)
            
            fig.show()

