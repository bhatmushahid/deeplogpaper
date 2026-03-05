import torch
import torch.nn as nn

# This simulates exactly what DeepLog does:
# Input: h=3 recent log keys
# Output: probability distribution over n=5 possible log keys
# Anomaly: if actual next key is NOT in top-g=2 predictions

# Simulated log key vocabulary (in real DeepLog: n=29 for HDFS)
n_log_keys = 5   # K = {k0, k1, k2, k3, k4}

# Simulated normal execution sequences (like parsed log keys)
# Think of these as: k2 -> k0 -> k3 -> k1 is a normal execution path
normal_sequences = [
    [2, 0, 3],  # window of h=3 keys
    [0, 3, 1],
    [3, 1, 4],
    [1, 4, 2],
    [4, 2, 0],
    [2, 0, 3],
    [0, 3, 1],
]
normal_labels = [1, 4, 2, 0, 3, 1, 4]  # next key in sequence

# Convert to tensors
X_train = torch.tensor(normal_sequences, dtype=torch.long)
y_train = torch.tensor(normal_labels, dtype=torch.long)

print("Training data (normal sequences only, like DeepLog):")
print(f"Input shape: {X_train.shape}")  # (7, 3)
print(f"Sample: {normal_sequences[0]} -> next key: {normal_labels[0]}")

# DeepLog-style LSTM model
class DeepLogLSTM(nn.Module):
    def __init__(self, n_keys, embed_dim, hidden_size, num_layers):
        super(DeepLogLSTM, self).__init__()
        self.n_keys = n_keys
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding: converts log key integer to dense vector
        # DeepLog uses one-hot; we use embedding (slight improvement)
        self.embedding = nn.Embedding(n_keys, embed_dim)
        
        # LSTM layers (L=2 like DeepLog default)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output layer: hidden_size -> n_keys (probability per key)
        self.fc = nn.Linear(hidden_size, n_keys)
    
    def forward(self, x):
        # x shape: (batch, seq_len) — integers representing log keys
        
        # Embed log keys
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        
        # LSTM forward
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(embedded, (h0, c0))
        
        # Take last time step output
        out = out[:, -1, :]    # (batch, hidden_size)
        out = self.fc(out)     # (batch, n_keys)
        return out

# Create model with DeepLog-like parameters
model = DeepLogLSTM(
    n_keys=n_log_keys,
    embed_dim=8,
    hidden_size=64,    # alpha=64 like DeepLog default
    num_layers=2       # L=2 like DeepLog default
)

print(f"\nDeepLog-style model created!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("\nTraining on NORMAL sequences only (DeepLog approach)...")
for epoch in range(200):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")

# Detection: top-g strategy (like DeepLog)
print("\n--- ANOMALY DETECTION (top-g=2 strategy) ---")
model.eval()
g = 2  # top-g candidates considered normal

test_cases = [
    ([2, 0, 3], 1, "NORMAL"),    # seen during training
    ([0, 3, 1], 4, "NORMAL"),    # seen during training
    ([2, 0, 3], 3, "ANOMALY"),   # unexpected key after this sequence
    ([1, 4, 2], 3, "ANOMALY"),   # unexpected key
]

with torch.no_grad():
    for seq, actual_key, expected_label in test_cases:
        x = torch.tensor([seq], dtype=torch.long)
        output = model(x)
        probs = torch.softmax(output, dim=1)
        top_g_keys = torch.topk(probs, g).indices[0].tolist()
        
        is_anomaly = actual_key not in top_g_keys
        result = "ANOMALY" if is_anomaly else "NORMAL"
        correct = "✓" if result == expected_label else "✗"
        
        print(f"{correct} Sequence {seq} -> actual key: k{actual_key}")
        print(f"  Top-{g} predicted keys: {['k'+str(k) for k in top_g_keys]}")
        print(f"  Detection: {result} (Expected: {expected_label})")
        print()