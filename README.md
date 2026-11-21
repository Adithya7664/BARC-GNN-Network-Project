# Contingency Analysis of Power Systems using PandaPower and ML

A Graph Neural Network (GNN)-based framework for N-1 contingency analysis in power systems. Replaces repeated full power flow solves with a trained GNN that classifies each single-element outage (line, transformer, generator, bus) as **critical** or **non-critical**.

---

## Results

| System | Load Scale | Accuracy | Precision | Recall | F1-Score |
|--------|-----------|----------|-----------|--------|----------|
| IEEE 118-bus | 1.25× | 93.56% | 95.79% | 82.73% | 88.78% |
| IEEE 57-bus | 1.05× | 84.62% | 80.56% | 87.88% | 84.06% |

**IEEE 118-bus Confusion Matrix:** TN=243, FP=4, FN=19, TP=91  
**IEEE 57-bus Confusion Matrix:** TN=63, FP=14, FN=8, TP=58

---

## Repository Structure

```
├── final_code.ipynb                            # Main notebook — IEEE 118-bus (load=1.25)
├── combined_models (3).ipynb                   # Main notebook — IEEE 57-bus (load=1.05)
├── gnn_dataset_all_ieee.pt                     # PyTorch GNN training dataset
├── best_model.pt                               # Trained GNN model weights
├── all_ieee_contingency_data.xlsx              # Full contingency simulation results
└── gnn_test_predictions_case118_load1.3.xlsx   # Test predictions on IEEE 118-bus
```

---

## Pipeline

```
IEEE Bus Systems (PandaPower)
        │
        ▼
N-1 Contingency Simulation
(lines, transformers, generators, buses taken out one at a time)
        │
        ▼
Data Preprocessing
(one-hot encoding, median imputation, normalization, critical flag)
        │
        ▼
Graph Construction (PyTorch Geometric)
Nodes = Buses  |  Edges = Lines + Transformers
Node features : voltage magnitude, outage flag, generator presence
Edge features : loading %, component type, outage flag
        │
        ▼
GNN Training (NNConv + AdamW + weighted CrossEntropyLoss)
        │
        ▼
Inference on unseen IEEE systems → Critical / Non-Critical
```

---

## Model Architecture

| Component | Detail |
|-----------|--------|
| GNN type | NNConv (edge-conditioned message passing), 3 layers |
| Pooling | Global mean + max, concatenated |
| Aux inputs | Contingency type (one-hot), load scaling factor |
| Optimizer | AdamW — lr=0.001, weight decay=0.01 |
| Loss | Weighted CrossEntropyLoss (class-balanced) |
| Epochs | 200 with early stopping on best val accuracy |
| Split | 80/20 stratified, seed=42 |

---

## Training Data

| Systems | Load scales |
|---------|-------------|
| case9, case14, case30, case39, case57, case118 | 1.0×, 1.25×, 1.50× |

---

## Dependencies

```
pandapower, pandas, numpy, matplotlib, networkx
torch, torch-geometric, scikit-learn, openpyxl
```

```bash
pip install pandapower pandas numpy matplotlib networkx torch torch-geometric scikit-learn openpyxl
```

---

## Usage

**IEEE 118-bus:**
```bash
jupyter notebook final_code.ipynb
```

**IEEE 57-bus:**
```bash
jupyter notebook "combined_models (3).ipynb"
```

Each notebook runs the full pipeline: contingency simulation → graph construction → GNN training → evaluation → Excel export.
