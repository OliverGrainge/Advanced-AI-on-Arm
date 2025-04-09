import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import display, clear_output
from pytorch_lightning.callbacks import Callback
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import requests
from torch.nn import TransformerEncoder, TransformerEncoderLayer, Embedding, Linear


class TinyShakespeareDataset(Dataset):
    def __init__(self, split='train', sequence_length=128):
        super().__init__()
        self.sequence_length = sequence_length
        
        # Download Shakespeare data if needed
        text_path = "tinyshakespeare.txt"
        if not os.path.exists(text_path):
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(text_path, 'w') as f:
                f.write(requests.get(url).text)
        
        # Load full text
        with open(text_path, 'r') as f:
            text = f.read()
        
        # Create character level vocabulary
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Split data
        n = len(text)
        train_split = int(n * 0.9)
        val_split = int(n * 0.95)
        
        if split == 'train':
            self.data = text[:train_split]
        elif split == 'valid':
            self.data = text[train_split:val_split]
        else:  # 'test'
            self.data = text[val_split:]
            
        self.encoded_data = [self.char_to_idx[c] for c in self.data]
    
    def __len__(self):
        return max(0, len(self.encoded_data) - self.sequence_length)
    
    def __getitem__(self, idx):
        # Get input sequence and target (next character prediction)
        inp = self.encoded_data[idx:idx+self.sequence_length]
        target = self.encoded_data[idx+1:idx+self.sequence_length+1]
        return torch.tensor(inp, dtype=torch.long), torch.tensor(target, dtype=torch.long)


class CharTransformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    ):
        super().__init__()
        self.pos_encoder = torch.nn.Embedding(1024, d_model)  # Position encoding
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.embedding = Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = Linear(d_model, vocab_size)
        
    def forward(self, src):
        seq_len = src.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=src.device).unsqueeze(0)
        pos_emb = self.pos_encoder(pos)
        
        # Token embeddings + positional embeddings
        src = self.embedding(src) + pos_emb
        
        # Apply transformer
        output = self.transformer_encoder(src)
        
        # Predict next token
        output = self.linear(output)
        return output


class TinyShakespeareModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size=None,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=64,
        sequence_length=128,
        num_workers=0,
        weight_decay=0.01
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Will be set during setup
        self.vocab_size = vocab_size
        
        self.model = None  # Will be initialized in setup after we know vocab_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.weight_decay = weight_decay

        # Track metrics history
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_perplexity_history = []
        
        # For validation metrics collection
        self.validation_step_outputs = []
        
        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        """Setup datasets for training and validation."""
        if stage == 'fit' or stage is None:
            self.train_dataset = TinyShakespeareDataset(
                split='train',
                sequence_length=self.sequence_length
            )
            
            self.val_dataset = TinyShakespeareDataset(
                split='valid',
                sequence_length=self.sequence_length
            )
            
            # Now we know the vocab size, initialize the model
            if self.vocab_size is None:
                self.vocab_size = self.train_dataset.vocab_size
                self.model = CharTransformer(
                    vocab_size=self.vocab_size,
                    d_model=self.hparams.d_model,
                    nhead=self.hparams.nhead,
                    num_layers=self.hparams.num_layers,
                    dim_feedforward=self.hparams.dim_feedforward,
                    dropout=self.hparams.dropout
                )
        
        if stage == 'test':
            self.val_dataset = TinyShakespeareDataset(
                split='test',
                sequence_length=self.sequence_length
            )
            if self.vocab_size is None:
                self.vocab_size = self.val_dataset.vocab_size
                self.model = CharTransformer(
                    vocab_size=self.vocab_size,
                    d_model=self.hparams.d_model,
                    nhead=self.hparams.nhead,
                    num_layers=self.hparams.num_layers,
                    dim_feedforward=self.hparams.dim_feedforward,
                    dropout=self.hparams.dropout
                )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        
        self.train_loss_history.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        perplexity = torch.exp(loss)
        
        # Store batch results
        self.validation_step_outputs.append({'loss': loss, 'perplexity': perplexity})
        return {'loss': loss, 'perplexity': perplexity}

    def on_validation_epoch_end(self):
        # Calculate average loss and perplexity for the epoch
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_perplexity = torch.stack([x['perplexity'] for x in self.validation_step_outputs]).mean()
        
        # Append to history
        self.val_loss_history.append(avg_loss.item())
        self.val_perplexity_history.append(avg_perplexity.item())
        
        # Log metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_perplexity', avg_perplexity, prog_bar=True)
        
        # Clear validation step outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        perplexity = torch.exp(loss)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_perplexity', perplexity, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Warmup + cosine schedule
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            ),
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return self.val_dataloader()
    
    def generate(self, seed_text, max_new_chars=100, temperature=1.0):
        """Generate text starting from seed_text"""
        if not hasattr(self, 'train_dataset'):
            raise ValueError("Model needs to be set up first")
        
        self.eval()
        dataset = self.train_dataset  # Just to access the char mapping
        
        # Convert seed text to indices
        chars = [dataset.char_to_idx[c] for c in seed_text]
        generated = list(chars)
        
        device = next(self.parameters()).device
        
        # Generate new characters
        with torch.no_grad():
            for _ in range(max_new_chars):
                # Prepare input sequence, keeping the last sequence_length tokens
                x = torch.tensor([generated[-self.sequence_length:]], dtype=torch.long, device=device)
                
                # Get model prediction
                logits = self(x)
                next_char_logits = logits[0, -1, :] / temperature
                
                # Apply temperature and sample
                probs = F.softmax(next_char_logits, dim=0)
                next_char_idx = torch.multinomial(probs, 1).item()
                
                # Add to generated sequence
                generated.append(next_char_idx)
        
        # Convert back to text
        generated_text = ''.join([dataset.idx_to_char[idx] for idx in generated])
        return generated_text


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
                        y=pl_module.val_loss_history,
                        name="Validation Loss",
                        line=dict(color='orange'),
                        visible=False
                    ),
                    go.Scatter(
                        x=val_steps,
                        y=pl_module.val_perplexity_history,
                        name="Validation Perplexity",
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
                    fig.add_trace(trace, secondary_y=(trace.name == "Validation Perplexity"))
            
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
                currentvalue={"prefix": "Training Loss Smoothing: "},
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
            fig.update_yaxes(title_text="Perplexity", secondary_y=True)
            
            fig.show()