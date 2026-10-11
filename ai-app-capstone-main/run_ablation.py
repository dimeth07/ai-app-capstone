# run_ablation.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split
from src.models.model import MLP

# --- 1. Settings ---
DEVICE = "cpu" # Use "cuda" if you have a GPU
SEED = 42

def set_seed(s=SEED):
    torch.manual_seed(s)
    np.random.seed(s)

def load_data(kind):
    if kind == "moons":
        X, y = make_moons(n_samples=1000, noise=0.25, random_state=SEED)
    else: # "circles"
        X, y = make_circles(n_samples=1000, noise=0.1, random_state=SEED)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# --- 2. Training Function ---
def train_model(model, X_train, y_train, epochs=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
    return model

# --- 3. The Experiment Runner ---
def run_experiment(dataset, hidden_sizes, use_relu):
    X, y = load_data(dataset)
    # We only need a train set for this quick ablation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    model = MLP(hidden_sizes=hidden_sizes).to(DEVICE)
    # THIS IS THE ABLATION: we change one thing at a time
    if not use_relu:
        # Replace ReLU with Identity (no activation)
        for i, layer in enumerate(model.net):
            if isinstance(layer, nn.ReLU):
                model.net[i] = nn.Identity()

    train_model(model, X_train, y_train)

    model.eval()
    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean().item()
        val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()
    return train_acc, val_acc

# --- 4. Configurations to Test ---
CONFIGS = [
    ("circles", (8,), False, "circles 2-8-2 no-act"),
    ("circles", (8,), True,  "circles 2-8-2 ReLU"),
    ("circles", (64,), True, "circles 2-64-2 ReLU"),
    ("moons",   (8,), False, "moons 2-8-2 no-act"),
    ("moons",   (8,), True,  "moons 2-8-2 ReLU"),
]

# --- 5. Run Everything ---
print(f"{'Configuration':22s} {'Train Acc':10s} {'Val Acc':10s}")
print("-" * 45)
set_seed(SEED)
for dataset, hidden, relu, name in CONFIGS:
    train_acc, val_acc = run_experiment(dataset, hidden, relu)
    print(f"{name:22s} {train_acc:10.3f} {val_acc:10.3f}")

print("\nDone. Copy these numbers into your experiments/ablation.md file.")