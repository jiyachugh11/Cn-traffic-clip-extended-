import os

import torch

from src.datasets.traffic_dataset import TrafficDataset
from src.models.proposed_traffic_clip import ProposedTrafficCLIP

# ---------------------------------------------------------

# Configuration

# ---------------------------------------------------------

DATA_PATH = "data/traffic_data.npz"

MODEL_PATH = "output/proposed_traffic_clip_best.pth"

NUM_CLASSES = 10

DEVICE = torch.device(
"cuda" if torch.cuda.is_available() else "cpu"
)

# ---------------------------------------------------------

# Prediction function

# ---------------------------------------------------------

@torch.no_grad()
def predict_sample(model, sample, device):

```
image = sample["image"].unsqueeze(0).to(device)

input_ids = sample["input_ids"].unsqueeze(0).to(device)

attention_mask = (
    sample["attention_mask"]
    .unsqueeze(0)
    .to(device)
)

stats = sample["stats"].unsqueeze(0).to(device)

# Open-set prediction
result = model.predict_open_set(
    image,
    input_ids,
    attention_mask,
    stats
)

prediction = result["prediction"].item()
confidence = result["confidence"].item()
entropy = result["entropy"].item()
known = result["known"].item()

return prediction, confidence, entropy, known
```

# ---------------------------------------------------------

# Main function

# ---------------------------------------------------------

def main():

```
print()
print("=" * 60)
print("Proposed TrafficCLIP - Prediction")
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
        "Prediction code is ready, but a real dataset "
        "sample is required."
    )

    return

# -----------------------------------------------------
# Check model
# -----------------------------------------------------

if not os.path.exists(MODEL_PATH):

    print()
    print("Trained model not found.")
    print("Expected:")
    print(MODEL_PATH)
    print()
    print(
        "Provide a trained checkpoint before prediction."
    )

    return

# -----------------------------------------------------
# Load dataset
# -----------------------------------------------------

print()
print("Loading dataset...")

dataset = TrafficDataset(
    npz_path=DATA_PATH,
    split="test"
)

if len(dataset) == 0:

    print("Dataset contains no samples.")
    return

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
    map_location=DEVICE,
    weights_only=True
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")

# -----------------------------------------------------
# Select sample
# -----------------------------------------------------

sample_index = 0

sample = dataset[sample_index]

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

print()
print("Running prediction...")

prediction, confidence, entropy, known = predict_sample(
    model,
    sample,
    DEVICE
)

# -----------------------------------------------------
# Display result
# -----------------------------------------------------

print()
print("=" * 60)
print("Prediction Result")
print("=" * 60)

print()

print("Sample index :", sample_index)

print(
    "Actual class :",
    sample["class_name"]
)

print(
    "Confidence   :",
    f"{confidence:.4f}"
)

print(
    "Entropy      :",
    f"{entropy:.4f}"
)

if known:

    print(
        "Prediction   :",
        prediction
    )

    print(
        "Status       : KNOWN"
    )

else:

    print(
        "Prediction   : UNKNOWN"
    )

    print(
        "Status       : ZERO-DAY / UNKNOWN"
    )

print()
print("=" * 60)
print("Prediction completed.")
print("=" * 60)
```

if **name** == "**main**":
main()
