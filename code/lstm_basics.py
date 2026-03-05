import torch
import torch.nn as nn

# A simple LSTM that predicts the next number in a sequence
# Example: given [1, 2, 3] → predict 4

# Step 1: Create dummy sequence data
# We'll predict the next number in sequences of 0-9
def make_sequence_data():
    sequences = []
    labels = []
    data = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    for i in range(len(data) - 3):
        seq = data[i:i+3]      # input: 3 numbers
        label = data[i+3]      # output: next number
        sequences.append(seq)
        labels.append(label)
    
    return sequences, labels

sequences, labels = make_sequence_data()
print("Sample input sequence:", sequences[0])  # [0, 1, 2]
print("Expected next number:", labels[0])       # 3

# Step 2: Convert to tensors
X = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  # shape: (7, 3, 1)
y = torch.tensor(labels, dtype=torch.long)                       # shape: (7,)

print("Input shape:", X.shape)   # (batch=7, seq_len=3, input_size=1)
print("Label shape:", y.shape)   # (7,)

# Step 3: Build a simple LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,   # size of each input element
            hidden_size=hidden_size, # size of hidden state
            num_layers=num_layers,   # number of stacked LSTM layers
            batch_first=True         # input shape is (batch, seq, feature)
        )
        
        # Final linear layer to produce output
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Forward pass through LSTM
        # out shape: (batch, seq_len, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Take output from the LAST time step only
        out = out[:, -1, :]  # shape: (batch, hidden_size)
        
        # Pass through linear layer
        out = self.fc(out)   # shape: (batch, output_size)
        return out

# Step 4: Create the model
model = SimpleLSTM(
    input_size=1,    # each input is 1 number
    hidden_size=16,  # 16 memory units
    output_size=10,  # predict one of 10 possible numbers (0-9)
    num_layers=2     # 2 stacked LSTM layers (like DeepLog's L=2)
)

print("\nModel architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# Step 5: Define loss and optimizer
criterion = nn.CrossEntropyLoss()   # same loss as DeepLog uses
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Step 6: Training loop
print("\nStarting training...")
for epoch in range(100):
    # Forward pass
    outputs = model(X)          # shape: (7, 10)
    loss = criterion(outputs, y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# Step 7: Test the model
print("\nTesting the model:")
model.eval()
with torch.no_grad():
    test_input = torch.tensor([[1, 2, 3]], dtype=torch.float32).unsqueeze(-1)
    output = model(test_input)
    probabilities = torch.softmax(output, dim=1)
    predicted = torch.argmax(probabilities, dim=1)
    print(f"Input sequence: [1, 2, 3]")
    print(f"Predicted next number: {predicted.item()}")
    print(f"Expected: 4")
    print(f"Top-3 predictions: {torch.topk(probabilities, 3).indices[0].tolist()}")