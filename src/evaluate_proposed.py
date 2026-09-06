import os

import torch
from torch.utils.data import DataLoader

from src.datasets.traffic_dataset import TrafficDataset
from src.models.proposed_traffic_clip import ProposedTrafficCLIP

# ---------------------------------------------------------

# Configuration

# ---------------------------------------------------------

DATA_PATH = "data/traffic_data.npz"

MODEL_PATH = "output/proposed_traffic_clip_best.pth"

BATCH_SIZE = 16

NUM_CLASSES = 10

DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------------

# Evaluation function

# ---------------------------------------------------------

@torch.no_grad()
def evaluate(model, dataloader, device):

```
model.eval()

total = 0
correct = 0

all_predictions = []
all_labels = []

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

    predictions = torch.argmax(logits, dim=1)

    correct += (
        predictions == labels
    ).sum().item()

    total += labels.size(0)

    all_predictions.extend(
        predictions.cpu().tolist()
    )

    all_labels.extend(
        labels.cpu().tolist()
    )

accuracy = correct / max(total, 1)

return accuracy, all_predictions, all_labels
```

# ---------------------------------------------------------

# Main evaluation function

# ---------------------------------------------------------

def main():

```
print()
print("=" * 60)
print("Proposed TrafficCLIP - Evaluation")
print("=" * 60)

print()
print("Device:", DEVICE)

# -----------------------------------------------------
# Check dataset
# -----------------------------------------------------

if not os.path.exists(DATA_PATH):

    print()
    print("Dataset file not found.")
    print("Expected:")
    print(DATA_PATH)
    print()
    print(
        "Evaluation code is ready, but the real dataset "
        "must be provided before evaluation."
    )

    return

# -----------------------------------------------------
# Check trained model
# -----------------------------------------------------

if not os.path.exists(MODEL_PATH):

    print()
    print("Trained model not found.")
    print("Expected:")
    print(MODEL_PATH)
    print()
    print(
        "Train the model first or provide the trained "
        "checkpoint."
    )

    return

# -----------------------------------------------------
# Load dataset
# -----------------------------------------------------

print()
print("Loading test dataset...")

test_dataset = TrafficDataset(
    npz_path=DATA_PATH,
    split="test"
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    "Test samples:",
    len(test_dataset)
)

# -----------------------------------------------------
# Create model
# -----------------------------------------------------

print()
print("Creating model...")

model = ProposedTrafficCLIP(
    num_classes=NUM_CLASSES,
    stats_input_dim=8
)

model = model.to(DEVICE)

# -----------------------------------------------------
# Load trained weights
# -----------------------------------------------------

print()
print("Loading trained model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)

print("Model loaded successfully.")

# -----------------------------------------------------
# Evaluate
# -----------------------------------------------------

print()
print("Evaluating model...")

accuracy, predictions, labels = evaluate(
    model,
    test_loader,
    DEVICE
)

# -----------------------------------------------------
# Display results
# -----------------------------------------------------

print()
print("=" * 60)
print("Evaluation Results")
print("=" * 60)

print()
print(
    f"Test Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Correct       : "
    f"{sum(p == y for p, y in zip(predictions, labels))}"
)

print(
    f"Incorrect     : "
    f"{sum(p != y for p, y in zip(predictions, labels))}"
)

print()
print("=" * 60)
print("Evaluation completed.")
print("=" * 60)
```

if **name** == "**main**":
main()
