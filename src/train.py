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

AUGMENTATION_STRENGTH = "strong"
USE_PROJECTION = True

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

def set_seed(seed=SEED):
    """
    Set random seeds for reproducible experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def get_device():
    """
    Select GPU if available, otherwise CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"Using GPU: {torch.cuda.get_device_name(0)}"
        )
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
    path,
    config=None
):
    """
    Save model, optimizer, training state, and configuration.
    """

    if config is None:
        config = {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
            "representation_dim": REPRESENTATION_DIM,
            "projection_hidden_dim": PROJECTION_HIDDEN_DIM,
            "projection_dim": PROJECTION_DIM,
            "augmentation_strength": AUGMENTATION_STRENGTH,
            "use_projection": USE_PROJECTION
        }

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "config": config
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# Training: One Epoch
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    verbose=True
):
    """
    Train the model for one epoch.

    Each batch contains two augmented views of the
    same images.

    The model produces:

        h = encoder representation
        z = projection representation

    The contrastive loss is computed using z.
    """

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

        if verbose and (batch_idx + 1) % 50 == 0:
            print(
                f"Epoch [{epoch}] "
                f"Batch [{batch_idx + 1}/{num_batches}] "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / num_batches

    return average_loss


# ============================================================
# Reusable Training Function
# ============================================================

def train_model(
    temperature=TEMPERATURE,
    augmentation_strength=AUGMENTATION_STRENGTH,
    use_projection=USE_PROJECTION,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    representation_dim=REPRESENTATION_DIM,
    projection_hidden_dim=PROJECTION_HIDDEN_DIM,
    projection_dim=PROJECTION_DIM,
    seed=SEED,
    save_checkpoints=False,
    best_model_path=None,
    last_model_path=None,
    history_path=None,
    verbose=True
):
    """
    Reusable contrastive-learning training function.

    Parameters
    ----------
    temperature : float
        Temperature used by InfoNCE / NT-Xent.

    augmentation_strength : str
        "weak" or "strong".

    use_projection : bool
        Whether the projection head is used.

    epochs : int
        Number of training epochs.

    batch_size : int
        Training batch size.

    save_checkpoints : bool
        Whether checkpoints should be saved.

    Returns
    -------
    model : ContrastiveModel
        Trained model.

    history : list
        Training loss history.

    device : torch.device
        Device used for training.
    """

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(seed)

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    if save_checkpoints:
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

    if verbose:
        print(f"Device: {device}")

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    if verbose:
        print()
        print("Loading CIFAR-10...")

    train_loader = get_contrastive_dataloader(
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_strength=augmentation_strength
    )

    if verbose:
        print(
            f"Training samples: "
            f"{len(train_loader.dataset)}"
        )

        print(
            f"Batches per epoch: "
            f"{len(train_loader)}"
        )

        print(
            f"Augmentation: "
            f"{augmentation_strength}"
        )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if verbose:
        print()
        print("Creating model...")

        print(
            f"Projection head: "
            f"{'enabled' if use_projection else 'disabled'}"
        )

    model = ContrastiveModel(
        representation_dim=representation_dim,
        projection_hidden_dim=projection_hidden_dim,
        projection_dim=projection_dim,
        use_projection=use_projection
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = NTXentLoss(
        temperature=temperature
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # --------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------

    config = {
        "seed": seed,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "representation_dim": representation_dim,
        "projection_hidden_dim": projection_hidden_dim,
        "projection_dim": projection_dim,
        "augmentation_strength": augmentation_strength,
        "use_projection": use_projection
    }

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = []

    best_loss = float("inf")

    if verbose:
        print()
        print("=" * 60)
        print("TRAINING")
        print("=" * 60)

    for epoch in range(1, epochs + 1):

        if verbose:
            print()
            print(
                f"Epoch {epoch}/{epochs}"
            )

        average_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            verbose=verbose
        )

        if verbose:
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

        if save_checkpoints:

            current_last_path = (
                last_model_path
                if last_model_path is not None
                else LAST_MODEL_PATH
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=average_loss,
                path=current_last_path,
                config=config
            )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if average_loss < best_loss:

            best_loss = average_loss

            if save_checkpoints:

                current_best_path = (
                    best_model_path
                    if best_model_path is not None
                    else BEST_MODEL_PATH
                )

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    loss=average_loss,
                    path=current_best_path,
                    config=config
                )

                if verbose:
                    print(
                        f"New best model saved "
                        f"(loss={best_loss:.4f})"
                    )

    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    if save_checkpoints:

        current_history_path = (
            history_path
            if history_path is not None
            else HISTORY_PATH
        )

        with open(
            current_history_path,
            "w"
        ) as f:

            json.dump(
                history,
                f,
                indent=4
            )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return model, history, device


# ============================================================
# Main Training Run
# ============================================================

def main():

    print("=" * 60)
    print("CONTRASTIVE LEARNING TRAINING")
    print("=" * 60)

    model, history, device = train_model(
        temperature=TEMPERATURE,
        augmentation_strength=AUGMENTATION_STRENGTH,
        use_projection=USE_PROJECTION,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        representation_dim=REPRESENTATION_DIM,
        projection_hidden_dim=PROJECTION_HIDDEN_DIM,
        projection_dim=PROJECTION_DIM,
        seed=SEED,
        save_checkpoints=True,
        best_model_path=BEST_MODEL_PATH,
        last_model_path=LAST_MODEL_PATH,
        history_path=HISTORY_PATH,
        verbose=True
    )

    best_loss = min(
        item["loss"]
        for item in history
    )

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


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
