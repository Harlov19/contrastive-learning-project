import os
import json
import random
import numpy as np

import torch
import torch.optim as optim

from dataset import get_contrastive_dataloader
from model import ContrastiveModel
from loss import NTXentLoss


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 128
NUM_WORKERS = 2

EPOCHS = 3

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

TEMPERATURE = 0.5

REPRESENTATION_DIM = 128
PROJECTION_HIDDEN_DIM = 128
PROJECTION_DIM = 64

CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pt"
)

LAST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    "last_model.pt"
)

HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    "training_history.json"
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    else:

        device = torch.device("cpu")

        print("CUDA not available. Using CPU.")

    return device


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path
):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,

        "config": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
            "representation_dim": REPRESENTATION_DIM,
            "projection_hidden_dim": PROJECTION_HIDDEN_DIM,
            "projection_dim": PROJECTION_DIM
        }
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch
):

    model.train()

    total_loss = 0.0

    num_batches = len(loader)

    for batch_idx, (view_1, view_2, _) in enumerate(loader):

        view_1 = view_1.to(device)
        view_2 = view_2.to(device)

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        _, z1 = model(view_1)
        _, z2 = model(view_2)

        # ----------------------------------------------------
        # Contrastive loss
        # ----------------------------------------------------

        loss = criterion(z1, z2)

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (batch_idx + 1) % 50 == 0:

            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx + 1}/{num_batches}] "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / num_batches

    return average_loss


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("CONTRASTIVE LEARNING TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print()
    print("Loading CIFAR-10...")

    train_loader = get_contrastive_dataloader(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Batches per epoch: "
        f"{len(train_loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("Creating model...")

    model = ContrastiveModel(
        representation_dim=REPRESENTATION_DIM,
        projection_hidden_dim=PROJECTION_HIDDEN_DIM,
        projection_dim=PROJECTION_DIM
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = NTXentLoss(
        temperature=TEMPERATURE
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_loss = float("inf")

    print()
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):

        print()
        print(f"Epoch {epoch}/{EPOCHS}")

        average_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch
        )

        print(
            f"Epoch {epoch} average loss: "
            f"{average_loss:.4f}"
        )

        history.append({
            "epoch": epoch,
            "loss": average_loss
        })

        # ----------------------------------------------------
        # Save last checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=average_loss,
            path=LAST_MODEL_PATH
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if average_loss < best_loss:

            best_loss = average_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=average_loss,
                path=BEST_MODEL_PATH
            )

            print(
                f"New best model saved "
                f"(loss={best_loss:.4f})"
            )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    with open(
        HISTORY_PATH,
        "w"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(
        f"Best loss: {best_loss:.4f}"
    )

    print(
        f"Best model: {BEST_MODEL_PATH}"
    )

    print(
        f"Training history: {HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()