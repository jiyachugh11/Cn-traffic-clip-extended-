import os
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.datasets.traffic_dataset import TrafficDataset
from src.models.proposed_traffic_clip import ProposedTrafficCLIP

logging.basicConfig(
level=logging.INFO,
format="%(asctime)s - %(message)s"
)

# ---------------------------------------------------------

# Configuration

# ---------------------------------------------------------

DATA_PATH = "data/traffic_data.npz"
OUTPUT_DIR = "output"

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 10

DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------------

# Training function

# ---------------------------------------------------------

def train_one_epoch(model, dataloader, criterion, optimizer, device):

```
model.train()

total_loss = 0.0
correct = 0
total = 0

for batch in dataloader:

    images = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    stats = batch["stats"].to(device)
    labels = batch["label"].to(device)

    # Forward pass
    logits = model(
        images,
        input_ids,
        attention_mask,
        stats
    )

    # Classification loss
    loss = criterion(logits, labels)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

    predictions = torch.argmax(logits, dim=1)

    correct += (predictions == labels).sum().item()
    total += labels.size(0)

average_loss = total_loss / max(len(dataloader), 1)
accuracy = correct / max(total, 1)

return average_loss, accuracy
```

# ---------------------------------------------------------

# Validation function

# ---------------------------------------------------------

@torch.no_grad()
def validate(model, dataloader, criterion, device):

```
model.eval()

total_loss = 0.0
correct = 0
total = 0

for batch in dataloader:

    images = batch["image"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    stats = batch["stats"].to(device)
    labels = batch["label"].to(device)

    logits = model(
        images,
        input_ids,
        attention_mask,
        stats
    )

    loss = criterion(logits, labels)

    total_loss += loss.item()

    predictions = torch.argmax(logits, dim=1)

    correct += (predictions == labels).sum().item()
    total += labels.size(0)

average_loss = total_loss / max(len(dataloader), 1)
accuracy = correct / max(total, 1)

return average_loss, accuracy
```

# ---------------------------------------------------------

# Main training function

# ---------------------------------------------------------

def main():

```
print()
print("=" * 60)
print("Proposed TrafficCLIP - Training")
print("=" * 60)

print()
print("Device:", DEVICE)

# -----------------------------------------------------
# Check dataset
# -----------------------------------------------------

if not os.path.exists(DATA_PATH):
    print()
    print("Dataset file not found.")
    print("Expected dataset:")
    print(DATA_PATH)
    print()
    print("Training code is ready, but the real dataset")
    print("must be provided before training can be started.")
    return

# -----------------------------------------------------
# Create dataset
# -----------------------------------------------------

print()
print("Loading dataset...")

dataset = TrafficDataset(
    npz_path=DATA_PATH,
    split="train"
)

validation_dataset = TrafficDataset(
    npz_path=DATA_PATH,
    split="val"
)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training samples   :", len(dataset))
print("Validation samples :", len(validation_dataset))

# -----------------------------------------------------
# Create model
# -----------------------------------------------------

print()
print("Creating ProposedTrafficCLIP model...")

model = ProposedTrafficCLIP(
    num_classes=NUM_CLASSES,
    stats_input_dim=8
)

model = model.to(DEVICE)

# -----------------------------------------------------
# Loss function
# -----------------------------------------------------

criterion = nn.CrossEntropyLoss()

# -----------------------------------------------------
# Optimizer
# -----------------------------------------------------

trainable_parameters = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_parameters,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

# -----------------------------------------------------
# Create output directory
# -----------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

best_validation_accuracy = 0.0

print()
print("Starting training...")
print()

# -----------------------------------------------------
# Training loop
# -----------------------------------------------------

for epoch in range(NUM_EPOCHS):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        DEVICE
    )

    validation_loss, validation_accuracy = validate(
        model,
        validation_loader,
        criterion,
        DEVICE
    )

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {validation_loss:.4f} "
        f"| Val Acc: {validation_accuracy:.4f}"
    )

    # -------------------------------------------------
    # Save best model
    # -------------------------------------------------

    if validation_accuracy > best_validation_accuracy:

        best_validation_accuracy = validation_accuracy

        checkpoint_path = os.path.join(
            OUTPUT_DIR,
            "proposed_traffic_clip_best.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "validation_accuracy": validation_accuracy
            },
            checkpoint_path
        )

        print(
            f"Best model saved to: {checkpoint_path}"
        )

print()
print("=" * 60)
print("Training completed.")
print(f"Best validation accuracy: {best_validation_accuracy:.4f}")
print("=" * 60)
```

if **name** == "**main**":
main()
