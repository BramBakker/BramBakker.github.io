import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, T5EncoderModel

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------
INPUT_FOLDER = Path("csv_files")
TEXT_COLUMN = "text"
LINE_COLUMN = "line"  # Column containing the original line numbers
MODEL_PATH = "byt5_stylometry_encoder.pt"
MAX_LINE_LENGTH = 120
LINES_PER_DOC = 500  # Number of points to sample per document group
OUTPUT_JSON_PATH = "stylometry_data.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------
# MODEL ARCHITECTURE
# --------------------------------------------------------------------------
class ByT5ContrastiveEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained("google/byt5-small")
        self.proj = nn.Sequential(
            nn.Linear(self.encoder.config.d_model, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
        pooled = (hidden_states * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        z = self.proj(pooled)
        return F.normalize(z, dim=1)


# --------------------------------------------------------------------------
# DATA LOADING & CONTEXT EXTRACTION
# --------------------------------------------------------------------------
def load_eval_subset(folder: Path, text_col: str, line_col: str, lines_per_doc: int = 5):
    known_records = []
    questioned_records = []

    for csv_file in sorted(folder.glob("*.csv")):
        stem = csv_file.stem.lower()
        # Classify the document type based on the file name
        if "cronycke_van_brabant" in stem:
            doc_type = "known"
        elif "voortzetting" in stem:
            doc_type = "questioned"
        else:
            continue  # Skip any unrelated files
            
        df = pd.read_csv(csv_file)
        if text_col not in df.columns:
            continue
            
        # Filter down to valid, non-empty text rows cleanly
        valid_df = df[df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")].copy()
        if valid_df.empty:
            continue
            
        # Sample random row indices
        sampled_indices = random.sample(list(valid_df.index), k=min(lines_per_doc, len(valid_df)))
              
        for idx in sampled_indices:
            # Extract target line information
            target_text = str(df.loc[idx, text_col]).strip()
            if line_col in df.columns:
                raw_line = df.loc[idx, line_col]
                try:
                    # Try to keep it as an integer if it's a normal number
                    line_no = int(raw_line)
                except (ValueError, TypeError):
                    # If it has brackets like [6]4574, keep it exactly as a string
                    line_no = str(raw_line).strip()
            else:
                line_no = idx + 1
                
            # Grab a local multi-line context window (current line + next 3 lines) matching your JSON style
            context_window = df.loc[idx : idx + 3, text_col].dropna().astype(str).str.strip().tolist()
            
            record = {
                "doc_type": doc_type,
                "line_no": line_no,
                "target_text": target_text,
                "lines": context_window
            }
            
            if doc_type == "known":
                known_records.append(record)
            else:
                questioned_records.append(record)
                
    # Balance or combine target samples
    combined_records = known_records[:lines_per_doc] + questioned_records[:lines_per_doc]
    return combined_records

# --------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------
def main():
    random.seed(42)
    
    print("Step 1: Loading text segments and local context windows...")
    records = load_eval_subset(INPUT_FOLDER, TEXT_COLUMN, LINE_COLUMN, LINES_PER_DOC)
    print(f"Loaded {len(records)} sample nodes across the dataset categories.")
    
    if not records:
        print("Error: No matching CSV files found. Check your file names and INPUT_FOLDER path.")
        return

    print("\nStep 2: Instantiating Model & Loading Trained Weights...")
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    model = ByT5ContrastiveEncoder(proj_dim=128).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Successfully loaded trained weights!")
    except FileNotFoundError:
        print(f"Warning: '{MODEL_PATH}' not found. Using baseline weights instead.")

    model.eval()
    
    print("\nStep 3: Embedding targeted lines...")
    batch_lines = [r["target_text"] for r in records]
    
    with torch.no_grad():
        inputs = tokenizer(
            batch_lines,
            padding=True,
            truncation=True,
            max_length=MAX_LINE_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)
        
        embeddings = model(inputs.input_ids, inputs.attention_mask).cpu().numpy()

    print("\nStep 4: Running Dimensionality Reduction (PCA)...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    # --------------------------------------------------------------------------
    # CONVERT TO TARGET SCHEMA AND SAVE JSON
    # --------------------------------------------------------------------------
    print("\nStep 5: Structuring data into target JSON format...")
    
    json_points = []
    known_counter = 1
    questioned_counter = 1
    
    for idx, record in enumerate(records):
        doc_type = record["doc_type"]
        
        # Generate alternating IDs (k1, k2... or q1, q2...)
        if doc_type == "known":
            point_id = f"k{known_counter}"
            known_counter += 1
        else:
            point_id = f"q{questioned_counter}"
            questioned_counter += 1
            
        point_data = {
            "id": point_id,
            "doc": doc_type,
            "x": round(float(coords[idx, 0]), 4),
            "y": round(float(coords[idx, 1]), 4),
            "line_no": record["line_no"],
            "match_index": 1,
            "lines": record["lines"]
        }
        json_points.append(point_data)
        
    final_output = {
        "note": "Line-level stylometric projections for Cronycke van Brabant and Brabantsche Yeesten.",
        "known_label": "Cronycke van Brabant",
        "questioned_label": "Brabantsche Yeesten",
        "points": json_points
    }
    
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as json_file:
        json.dump(final_output, json_file, indent=1, ensure_ascii=False)
        
    print(f"Success! High-dimensional points converted and saved to '{OUTPUT_JSON_PATH}'.")

if __name__ == "__main__":
    main()