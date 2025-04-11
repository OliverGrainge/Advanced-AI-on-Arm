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
from datasets import load_dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import TemplateProcessing
import tempfile
import shutil
import numpy as np 

class GPT2TokenizerWrapper:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Ensure the tokenizer uses the GPT2 padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def encode(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        """Encode text to token IDs"""
        encoding = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length
        )
        return encoding
    
    def decode(self, ids):
        """Decode token IDs to text"""
        return self.tokenizer.decode(ids)
    
    def get_vocab_size(self):
        """Get the vocabulary size"""
        return len(self.tokenizer)

class WikiText2Dataset(Dataset):
    def __init__(self, split='train', tokenizer=None, sequence_length=128, max_samples=None, vocab_size=None):
        self.sequence_length = sequence_length
        self.dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # Limit dataset size if max_samples is provided
        if max_samples is not None and max_samples > 0:
            print(f"Limiting {split} dataset to {max_samples} samples")
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        # Use provided tokenizer or create a new one
        if tokenizer is None:
            # Create a GPT2 tokenizer
            self.tokenizer = GPT2TokenizerWrapper()
        else:
            self.tokenizer = tokenizer

        # Process text in chunks to avoid sequence length issues
        all_tokens = []
        for text in self.dataset['text']:
            # Skip empty lines
            if not text.strip():
                continue
            # Tokenize each text chunk separately
            tokenized = self.tokenizer.encode(text, return_tensors='pt', padding=False, truncation=True, max_length=1024)
            all_tokens.append(tokenized['input_ids'].squeeze())
        
        # Concatenate all tokenized chunks
        self.tokens = torch.cat(all_tokens)

    def __len__(self):
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx+self.sequence_length]
        target_ids = self.tokens[idx+1:idx+self.sequence_length+1]
        return input_ids, target_ids


class AutoRegressiveTransformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size=8192,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.pos_encoder = torch.nn.Embedding(1024, d_model)  # Position encoding
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.embedding = Embedding(self.vocab_size, d_model)
        self.d_model = d_model
        self.linear = Linear(d_model, self.vocab_size)
        
        
    def forward(self, src):
        src = src.long()
        seq_len = src.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=src.device).unsqueeze(0)
        pos_emb = self.pos_encoder(pos)
        src = self.embedding(src) + pos_emb

        # Create a causal mask (upper triangular with -infty)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=src.device), diagonal=1)
        
        output = self.transformer_encoder(src, mask=mask)
        output = self.linear(output)
        return output


class WikiText2Module(pl.LightningModule):
    def __init__(
        self,
        vocab_size=50257,  # GPT2 tokenizer vocab size
        d_model=128,
        nhead=8,
        num_layers=8,
        dim_feedforward=1024,
        dropout=0.1,
        learning_rate=1e-4,
        batch_size=64,
        sequence_length=128,
        num_workers=15,
        weight_decay=0.01,
        max_train_samples=10000,
        max_val_samples=2000,
        max_test_samples=2000
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab_size = vocab_size
        self.model = None  # Will be initialized in setup after we know vocab_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.weight_decay = weight_decay
        self.tokenizer = GPT2TokenizerWrapper()
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples
        self.max_test_samples = max_test_samples

        # Track metrics history
        self.train_loss_history = []
        self.val_loss_history = []
        self.val_perplexity_history = []
        
        # For validation metrics collection
        self.validation_step_outputs = []
        
        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None

        self.model = AutoRegressiveTransformer(
            d_model=self.hparams.d_model,
            nhead=self.hparams.nhead,
            num_layers=self.hparams.num_layers,
            dim_feedforward=self.hparams.dim_feedforward,
            dropout=self.hparams.dropout,
            vocab_size=self.vocab_size,
        )

    def setup(self, stage=None):
        """Setup datasets for training and validation."""
        if stage == 'fit' or stage is None:
            # Initialize the GPT2 tokenizer if not already done
            
            # Now create the actual datasets with the tokenizer
            self.train_dataset = WikiText2Dataset(
                split='train',
                tokenizer=self.tokenizer,
                sequence_length=self.sequence_length,
                max_samples=self.max_train_samples,
            )
            
            self.val_dataset = WikiText2Dataset(
                split='validation', 
                tokenizer=self.tokenizer,
                sequence_length=self.sequence_length,
                max_samples=self.max_val_samples,
            )
        
        elif stage == 'test':
            # For testing, we need to load the tokenizer from the training stage
            if self.tokenizer is None:
                self.tokenizer = GPT2TokenizerWrapper()
                
            self.test_dataset = WikiText2Dataset(
                split='test',
                tokenizer=self.tokenizer,
                sequence_length=self.sequence_length,
                max_samples=self.max_test_samples,
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # Convert target tensor to Long type
        y = y.long()
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
        
        self.train_loss_history.append(loss.item())
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        # Convert target tensor to Long type
        y = y.long()
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
        # Convert target tensor to Long type
        y = y.long()
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
        #scheduler = {
        #    "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
        #        optimizer,
        #        T_max=self.trainer.max_epochs,
        #        eta_min=1e-6
        #    ),
        #    "interval": "epoch",
        #    "frequency": 1
        #}
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler}

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
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def generate(self, seed_text, max_new_tokens=100, temperature=1.0):
        """Generate text starting from seed_text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Run training first.")
        
        self.eval()
        
        # Convert seed text to indices
        encoded = self.tokenizer.encode(seed_text)
        input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long, device=next(self.parameters()).device)
        
        # Generate new tokens
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Prepare input sequence, keeping the last sequence_length tokens
                if input_ids.size(0) > self.sequence_length:
                    input_ids = input_ids[-self.sequence_length:]
                
                # Add batch dimension
                x = input_ids.unsqueeze(0)
                
                # Get model prediction
                logits = self(x)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply temperature and sample
                probs = F.softmax(next_token_logits, dim=0)
                next_token_idx = torch.multinomial(probs, 1).item()
                
                # Add to generated sequence
                input_ids = torch.cat([input_ids, torch.tensor([next_token_idx], device=input_ids.device)])
        
        # Convert back to text
        generated_text = self.tokenizer.decode(input_ids.tolist())
        return generated_text




class PlotlyCallback(Callback):
    def __init__(self, val_check_interval=1):
        self.val_check_interval=val_check_interval

        # Track validation points in terms of training progress
        self.validation_points = []
    
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
    
    def on_validation_start(self, trainer, pl_module):
        # Calculate current progress in terms of epochs (can be fractional)
        current_epoch_progress = trainer.current_epoch
        if hasattr(trainer, 'global_step') and hasattr(trainer, 'num_training_batches'):
            # Add the fraction of current epoch completed
            current_epoch_progress += (trainer.global_step % trainer.num_training_batches) / trainer.num_training_batches
        
        # Store the validation point
        self.validation_points.append(current_epoch_progress)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if not pl_module.train_loss_history:
            return
        
        # Check if we should plot based on validation count rather than epoch
        if len(pl_module.val_loss_history) > 0:
            # Calculate steps per epoch
            steps_per_epoch = len(pl_module.train_dataloader())
            
            # Create x-axis values
            train_steps = [(i/steps_per_epoch) for i in range(len(pl_module.train_loss_history))]
            
            # Use the tracked validation points instead of assuming 1 per epoch
            # If validation_points is empty or not long enough, fall back to range-based approach
            if len(self.validation_points) == len(pl_module.val_loss_history):
                val_steps = self.validation_points
            else:
                # Fall back to original behavior with range
                val_steps = list(range(len(pl_module.val_loss_history)))
            
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
                        x=torch.arange(len(pl_module.val_loss_history)) * self.val_check_interval,
                        y=pl_module.val_loss_history,
                        name="Validation Loss",
                        line=dict(color='orange'),
                        visible=False
                    ),
                    go.Scatter(
                        x=torch.arange(len(pl_module.val_loss_history)) * self.val_check_interval,
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