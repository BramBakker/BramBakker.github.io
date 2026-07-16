import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
INPUT_FOLDER = Path("csv_files")
TEXT_COLUMN = "text"
AUTHOR_COLUMN = "author"

# P-K Sampling Parameters (Controls batch sizing)
AUTHORS_PER_BATCH = 3   # "P" - number of unique authors per batch
LINES_PER_AUTHOR = 5   # "K" - number of lines sampled per author
MAX_LINE_LENGTH = 120   # Crop lines longer than this to protect VRAM

EPOCHS = 10
BATCHES_PER_EPOCH = 50
LEARNING_RATE = 0.001    # Lower learning rate for tuning transformers
TEMPERATURE = 0.1
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --------------------------------------------------------------------------
# DATA PROCESSING & LABELING
# --------------------------------------------------------------------------
def load_rijm_corpus(folder: Path, text_col: str, author_col: str):
    all_lines = []
    all_labels = []
    
    for csv_file in sorted(folder.glob("*.csv")):
        df = pd.read_csv(csv_file)
        if text_col not in df.columns:
            continue
            
        # Determine the file's primary author label
        raw_author = "Onbekend"
        if author_col in df.columns and len(df[author_col].dropna()) > 0:
            raw_author = str(df[author_col].dropna().iloc[0]).strip()
            
        # Isolate anonymous items so they don't accidentally merge
        if raw_author.lower() in ["onbekend", "unknown", "nan", ""]:
            author_label = f"Onbekend_{csv_file.stem}"
        else:
            author_label = raw_author
            
        # Harvest every single valid line from the text column
        lines = df[text_col].dropna().astype(str).str.strip().tolist()
        lines = [l for l in lines if len(l) > 0] # drop blank rows
        
        all_lines.extend(lines)
        all_labels.extend([author_label] * len(lines))
        
    return all_lines, all_labels

# --------------------------------------------------------------------------
# SAMPLER FOR CONTRASTIVE BATCHES
# --------------------------------------------------------------------------
class ByT5PKSampler:
    def __init__(self, labels: np.ndarray, p: int, k: int):
        self.p = p
        self.k = k
        self.label_to_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.label_to_indices[lbl].append(idx)
            
        # Ensure classes have at least 2 lines to provide a valid positive pair
        self.eligible_labels = [lbl for lbl, idxs in self.label_to_indices.items() if len(idxs) >= 2]
        
    def sample_batch(self, rng: random.Random) -> list[int]:
        chosen_labels = rng.sample(self.eligible_labels, k=min(self.p, len(self.eligible_labels)))
        batch_indices = []
        for lbl in chosen_labels:
            idxs = self.label_to_indices[lbl]
            k_sampled = rng.sample(idxs, k=min(self.k, len(idxs)))
            batch_indices.extend(k_sampled)
        return batch_indices

# --------------------------------------------------------------------------
# MODEL ARCHITECTURE WITH(OUT) LAYER FREEZING
# --------------------------------------------------------------------------
class ByT5ContrastiveEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        # Load only the encoder part of ByT5-Small (approx. 120M parameters)
        self.encoder = T5EncoderModel.from_pretrained("google/byt5-small")
        
        # Projection head to keep pre-trained weights from collapsing during contrastive training
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, input_ids, attention_mask):
        # ByT5 outputs shape: (batch_size, seq_len, hidden_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Mean Pooling: Average character representations, ignoring padded characters
        mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        pooled = (hidden_states * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        
        # Project to contrastive embedding space and L2 normalize
        z = self.proj(pooled)
        return F.normalize(z, dim=1)

# --------------------------------------------------------------------------
# CONTRASTIVE LOSS FUNCTION
# --------------------------------------------------------------------------
def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    device = embeddings.device
    sim = embeddings @ embeddings.T / temperature
    labels = labels.view(-1, 1)
    
    pos_mask = (labels == labels.T).float() - torch.eye(len(labels), device=device)
    
    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    
    exp_logits = torch.exp(logits) * (1 - torch.eye(len(labels), device=device))
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    
    n_pos = pos_mask.sum(dim=1)
    valid = n_pos > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    mean_log_prob = (pos_mask * log_prob).sum(dim=1)[valid] / n_pos[valid]
    return -mean_log_prob.mean()

# --------------------------------------------------------------------------
# EXECUTION PIPELINE
# --------------------------------------------------------------------------
def main():
    set_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)
    
    print("Loading manuscripts from folder...")
    lines, string_labels = load_rijm_corpus(INPUT_FOLDER, TEXT_COLUMN, AUTHOR_COLUMN)
    
    unique_labels = sorted(list(set(string_labels)))
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    int_labels = np.array([label_to_id[lbl] for lbl in string_labels])
    
    print(f"Loaded {len(lines)} total lines across {len(unique_labels)} distinct author tracks.")
    
    # Initialize ByT5 structures
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = ByT5ContrastiveEncoder(proj_dim=128).to(DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    sampler = ByT5PKSampler(int_labels, AUTHORS_PER_BATCH, LINES_PER_AUTHOR)
    
    print("\nStarting ByT5 Training Sequence...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for _ in range(BATCHES_PER_EPOCH):
            batch_idx = sampler.sample_batch(rng)
            if len(batch_idx) < 4:
                continue
                
            batch_lines = [lines[i] for i in batch_idx]
            batch_y = torch.tensor(int_labels[batch_idx], dtype=torch.long, device=DEVICE)
            
            # Tokenize strings straight into UTF-8 token IDs
            inputs = tokenizer(
                batch_lines, 
                padding=True, 
                truncation=True, 
                max_length=MAX_LINE_LENGTH, 
                return_tensors="pt"
            ).to(DEVICE)
            
            z = model(inputs.input_ids, inputs.attention_mask)
            loss = supervised_contrastive_loss(z, batch_y, TEMPERATURE)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Average Loss: {epoch_loss / BATCHES_PER_EPOCH:.4f}")

    print("\nTraining complete! Saving tuned model parameters...")
    torch.save(model.state_dict(), "byt5_stylometry_encoder.pt")

if __name__ == "__main__":
    main()