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

class CustomBPETokenizer:
    def __init__(self, vocab_size=4096, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = None
        self.temp_dir = None
        
    def train(self, dataset):
        """Train the tokenizer on the dataset"""
        # Create a temporary directory to store the tokenizer files
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize a ByteLevelBPETokenizer
        tokenizer = ByteLevelBPETokenizer()
        
        # Prepare training files
        train_file = os.path.join(self.temp_dir, "train.txt")
        with open(train_file, "w", encoding="utf-8") as f:
            for item in dataset:
                f.write(item["text"] + "\n")
        
        # Train the tokenizer
        tokenizer.train(
            files=[train_file],
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["<s>", "</s>", "<unk>", "<pad>"]
        )
        
        # Save the tokenizer
        tokenizer.save(os.path.join(self.temp_dir, "tokenizer.json"))
        
        # Load the tokenizer
        self.tokenizer = tokenizer
        
        return self
    
    def encode(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        """Encode text to token IDs"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet. Call train() first.")
        
        # Encode the text
        encoding = self.tokenizer.encode(text)
        
        # Handle truncation
        if truncation and max_length is not None:
            # Get the truncated IDs
            ids = encoding.ids[:max_length]
        else:
            ids = encoding.ids
        
        # Convert to tensor if requested
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids)}
        
        return {"input_ids": ids}
    
    def decode(self, ids):
        """Decode token IDs to text"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet. Call train() first.")
        
        return self.tokenizer.decode(ids)
    
    def get_vocab_size(self):
        """Get the vocabulary size"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained yet. Call train() first.")
        
        return self.tokenizer.get_vocab_size()
    
    def __del__(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

class TinyShakespeareDataset(Dataset):
    def __init__(self, split='train', tokenizer=None, sequence_length=128, max_samples=None, vocab_size=8192, 
                 train_test_split=0.9, val_test_split=0.5):
        self.sequence_length = sequence_length
        
        # Download tiny Shakespeare dataset if needed
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        shakespeare_path = os.path.join(tempfile.gettempdir(), "tinyshakespeare.txt")
        
        if not os.path.exists(shakespeare_path):
            print(f"Downloading Tiny Shakespeare dataset...")
            response = requests.get(data_url)
            with open(shakespeare_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded to {shakespeare_path}")
            
        # Load the text
        with open(shakespeare_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Create dataset with text chunks
        dataset_items = []
        chunk_size = 1000  # Adjust this to control dataset size
        for i in range(0, len(text), chunk_size):
            dataset_items.append({"text": text[i:i+chunk_size]})
        
        # Use provided tokenizer or create a new one
        if tokenizer is None:
            # Create and train a new tokenizer
            self.tokenizer = CustomBPETokenizer(vocab_size=vocab_size)
            self.tokenizer.train(dataset_items)
        else:
            self.tokenizer = tokenizer
        
        # Split dataset based on requested split
        num_items = len(dataset_items)
        train_size = int(num_items * train_test_split)
        test_val_size = num_items - train_size
        val_size = int(test_val_size * val_test_split)
        test_size = test_val_size - val_size
        
        if split == 'train':
            dataset_items = dataset_items[:train_size]
        elif split == 'validation':
            dataset_items = dataset_items[train_size:train_size+val_size]
        else:  # test
            dataset_items = dataset_items[train_size+val_size:]
        
        # Limit dataset size if max_samples is provided
        if max_samples is not None and max_samples > 0:
            dataset_items = dataset_items[:min(max_samples, len(dataset_items))]

        # Process text in chunks to avoid sequence length issues
        all_tokens = []
        for item in dataset_items:
            # Tokenize each text chunk separately
            tokenized = self.tokenizer.encode(item["text"], return_tensors='pt', padding=False, truncation=True, max_length=1024)
            all_tokens.append(tokenized['input_ids'].squeeze())
        
        # Concatenate all tokenized chunks
        if all_tokens:
            self.tokens = torch.cat(all_tokens)
        else:
            # Handle empty dataset case
            self.tokens = torch.tensor([], dtype=torch.long)

    def __len__(self):
        if len(self.tokens) <= self.sequence_length:
            return 0
        return len(self.tokens) - self.sequence_length

    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx+self.sequence_length]
        target_ids = self.tokens[idx+1:idx+self.sequence_length+1]
        return input_ids, target_ids


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Combined projection for Q, K, V (more efficient)
        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model)
        
        # Output projection
        self.out_proj = torch.nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Combined linear projection for Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * d_model)
        
        # Split into Q, K, V and reshape for multi-head attention
        qkv = qkv.reshape(batch_size, seq_len, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, nhead, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each has shape (batch_size, nhead, seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch, nhead, seq_len, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask.unsqueeze(0).unsqueeze(0)  # Add to all batches and heads
        
        # Apply softmax and dropout
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, v)  # (batch, nhead, seq_len, head_dim)
        
        # Reshape and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout=0.1):
        super().__init__()
        self.linear1 = torch.nn.Linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = torch.nn.Linear(dim_feedforward, d_model)
        self.activation = torch.nn.GELU()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        # Self-attention layer
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        # Feedforward network
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        
        # Layer normalization
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # Feedforward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.norm2(x)
        
        return x


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
        self.d_model = d_model
        
        # Token embedding
        self.embedding = torch.nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoder = torch.nn.Embedding(1024, d_model)
        
        # Stack of transformer blocks
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final projection to vocabulary
        self.linear = torch.nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, src):
        src = src.long()
        batch_size, seq_len = src.shape
        
        # Token embeddings
        token_emb = self.embedding(src)
        
        # Positional embeddings
        pos = torch.arange(0, seq_len, dtype=torch.long, device=src.device).unsqueeze(0)
        pos_emb = self.pos_encoder(pos)
        
        # Combine token and positional embeddings
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Create causal mask (upper triangular with -inf)
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=src.device), diagonal=1)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask=mask)
        
        # Final projection to vocabulary
        output = self.linear(x)
        
        return output


class ShakespeareModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size=8192,
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
        self.tokenizer = None
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
            # Check if we're loading from a checkpoint
            if self.tokenizer is None:
                # Create a shared tokenizer for both train and validation
                self.tokenizer = CustomBPETokenizer(vocab_size=self.vocab_size)
                
                # Create a temporary dataset to train the tokenizer
                temp_dataset = TinyShakespeareDataset(
                    split='train',
                    tokenizer=None,
                    sequence_length=self.sequence_length,
                    vocab_size=self.vocab_size,
                    max_samples=self.max_train_samples
                )
                self.tokenizer = temp_dataset.tokenizer
            
            # Now create the actual datasets with the trained tokenizer
            self.train_dataset = TinyShakespeareDataset(
                split='train',
                tokenizer=self.tokenizer,
                sequence_length=self.sequence_length,
                max_samples=self.max_train_samples,
                vocab_size=self.vocab_size,
            )
            
            self.val_dataset = TinyShakespeareDataset(
                split='validation', 
                tokenizer=self.tokenizer,
                sequence_length=self.sequence_length,
                max_samples=self.max_val_samples,
                vocab_size=self.vocab_size,
            )
        
        elif stage == 'test':
            # For testing, we need to load the tokenizer from the training stage
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized. Run training first.")
                
            self.test_dataset = TinyShakespeareDataset(
                split='test',
                tokenizer=self.tokenizer,
                sequence_length=self.sequence_length,
                max_samples=self.max_test_samples,
                vocab_size=self.vocab_size,
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

    def on_save_checkpoint(self, checkpoint):
        """Save tokenizer state in the checkpoint"""
        if self.tokenizer is not None and self.tokenizer.tokenizer is not None:
            # Get the tokenizer.json content
            tokenizer_path = os.path.join(self.tokenizer.temp_dir, "tokenizer.json")
            if os.path.exists(tokenizer_path):
                with open(tokenizer_path, 'rb') as f:
                    tokenizer_content = f.read()
                checkpoint['tokenizer'] = tokenizer_content
    
    def on_load_checkpoint(self, checkpoint):
        """Load tokenizer from checkpoint"""
        if 'tokenizer' in checkpoint:
            # Initialize tokenizer
            self.tokenizer = CustomBPETokenizer(vocab_size=self.vocab_size)
            
            # Create a temporary directory for the tokenizer
            self.tokenizer.temp_dir = tempfile.mkdtemp()
            
            # Save the tokenizer content to the temporary directory
            tokenizer_path = os.path.join(self.tokenizer.temp_dir, "tokenizer.json")
            with open(tokenizer_path, 'wb') as f:
                f.write(checkpoint['tokenizer'])
            
            # Load the tokenizer from the saved file
            self.tokenizer.tokenizer = Tokenizer.from_file(tokenizer_path)


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
            
            # Use display and clear_output to replace the previous plot in Jupyter
            clear_output(wait=True)
            display(fig)