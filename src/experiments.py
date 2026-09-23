import os
import json

import torch
import torch.nn.functional as F

from train import train_model

from evaluate import (
    get_dataloaders,
    extract_embeddings,
    knn_predict,
    calculate_representation_geometry,
    select_geometry_samples,
)

from retrieval import (
    retrieve_neighbors,
    precision_at_k,
    retrieval_accuracy_at_k,
)


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = "./results/experiments"
CHECKPOINT_DIR = "./checkpoints/experiments"

SEED = 42

# Keep the experimental budget identical across conditions.
EXPERIMENT_EPOCHS = 2
EXPERIMENT_BATCH_SIZE = 128
EXPERIMENT_NUM_WORKERS = 2

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

REPRESENTATION_DIM = 128
PROJECTION_HIDDEN_DIM = 128
PROJECTION_DIM = 64

# ------------------------------------------------------------
# Evaluation subset
# ------------------------------------------------------------
# The same fixed indices are used for every experiment.
# This makes the controlled comparisons reproducible.

KNN_TRAIN_SAMPLES = 10_000
KNN_TEST_SAMPLES = 2_000

GEOMETRY_SAMPLES = 2_000

RETRIEVAL_DATABASE_SAMPLES = 10_000
RETRIEVAL_QUERIES = 1_000

KNN_K = 5
RETRIEVAL_K_VALUES = [1, 5]

EVALUATION_SEED = 42


# ============================================================
# UTILITIES
# ============================================================

def save_json(data, path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=4
        )


def print_experiment_header(name):
    print()
    print("=" * 70)
    print(name)
    print("=" * 70)


def save_experiment_checkpoint(
    model,
    history,
    config,
    path
):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True
    )

    checkpoint = {
        "epoch": history[-1]["epoch"],
        "loss": history[-1]["loss"],
        "model_state_dict": model.state_dict(),
        "config": config,
    }

    torch.save(
        checkpoint,
        path
    )


def create_fixed_indices(
    num_train,
    num_test
):
    """
    Create deterministic evaluation subsets.

    The same indices are reused for every experimental
    condition so that representation comparisons are
    performed on exactly the same samples.
    """

    generator = torch.Generator()
    generator.manual_seed(EVALUATION_SEED)

    train_indices = torch.randperm(
        50_000,
        generator=generator
    )[:num_train]

    test_indices = torch.randperm(
        10_000,
        generator=generator
    )[:num_test]

    return train_indices, test_indices


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model_representation(
    model,
    device,
    train_loader,
    test_loader,
    train_indices,
    test_indices,
):
    """
    Evaluate the encoder representation h.

    Metrics:
        - k-NN accuracy
        - representation geometry
        - retrieval Precision@1
        - retrieval Precision@5
        - retrieval Accuracy@1
        - retrieval Accuracy@5
    """

    print()
    print("-" * 70)
    print("EXTRACTING REPRESENTATIONS")
    print("-" * 70)

    # --------------------------------------------------------
    # Extract full evaluation embeddings
    # --------------------------------------------------------

    train_embeddings, train_labels = extract_embeddings(
        model=model,
        loader=train_loader,
        device=device
    )

    test_embeddings, test_labels = extract_embeddings(
        model=model,
        loader=test_loader,
        device=device
    )

    print(
        f"Train embeddings: {train_embeddings.shape}"
    )

    print(
        f"Test embeddings:  {test_embeddings.shape}"
    )

    # --------------------------------------------------------
    # Fixed subsets
    # --------------------------------------------------------

    knn_train_embeddings = (
        train_embeddings[train_indices]
    )

    knn_train_labels = (
        train_labels[train_indices]
    )

    knn_test_embeddings = (
        test_embeddings[test_indices]
    )

    knn_test_labels = (
        test_labels[test_indices]
    )

    # ========================================================
    # 1. k-NN
    # ========================================================

    print()
    print("-" * 70)
    print("k-NN EVALUATION")
    print("-" * 70)

    predictions = knn_predict(
        train_embeddings=knn_train_embeddings,
        train_labels=knn_train_labels,
        test_embeddings=knn_test_embeddings,
        k=KNN_K,
        query_batch_size=128
    )

    knn_accuracy = (
        predictions == knn_test_labels
    ).float().mean().item()

    print(
        f"k-NN accuracy: "
        f"{knn_accuracy * 100:.2f}%"
    )

    # ========================================================
    # 2. Representation geometry
    # ========================================================

    print()
    print("-" * 70)
    print("REPRESENTATION GEOMETRY")
    print("-" * 70)

    geometry_embeddings = (
        test_embeddings[test_indices]
    )

    geometry_labels = (
        test_labels[test_indices]
    )

    geometry = calculate_representation_geometry(
        embeddings=geometry_embeddings,
        labels=geometry_labels
    )

    print(
        f"Same-class cosine similarity: "
        f"{geometry['same_class_mean']:.4f}"
    )

    print(
        f"Different-class cosine similarity: "
        f"{geometry['different_class_mean']:.4f}"
    )

    print(
        f"Similarity gap: "
        f"{geometry['similarity_gap']:.4f}"
    )

    print(
        f"Average pairwise similarity: "
        f"{geometry['average_pairwise_similarity']:.4f}"
    )

    print(
        f"Average embedding variance: "
        f"{geometry['average_embedding_variance']:.6f}"
    )

    # ========================================================
    # 3. Retrieval
    # ========================================================

    print()
    print("-" * 70)
    print("RETRIEVAL EVALUATION")
    print("-" * 70)

    # Use a fixed database subset.
    database_embeddings = (
        test_embeddings[:RETRIEVAL_DATABASE_SAMPLES]
    )

    database_labels = (
        test_labels[:RETRIEVAL_DATABASE_SAMPLES]
    )

    # Fixed query indices within the database.
    query_generator = torch.Generator()
    query_generator.manual_seed(EVALUATION_SEED)

    query_indices = torch.randperm(
        RETRIEVAL_DATABASE_SAMPLES,
        generator=query_generator
    )[:RETRIEVAL_QUERIES]

    query_embeddings = (
        database_embeddings[query_indices]
    )

    query_labels = (
        database_labels[query_indices]
    )

    retrieval_results = {}

    for k in RETRIEVAL_K_VALUES:

        # Request k+1 because the query itself is
        # contained in the database and will be the
        # first nearest neighbor.
        _, _, retrieved_labels = retrieve_neighbors(
            query_embeddings=query_embeddings,
            database_embeddings=database_embeddings,
            database_labels=database_labels,
            k=k + 1
        )

        # Remove the query itself.
        retrieved_labels = (
            retrieved_labels[:, 1:]
        )

        precision = precision_at_k(
            retrieved_labels=retrieved_labels,
            query_labels=query_labels,
            k=k
        )

        accuracy = retrieval_accuracy_at_k(
            retrieved_labels=retrieved_labels,
            query_labels=query_labels,
            k=k
        )

        retrieval_results[str(k)] = {
            "precision": precision,
            "accuracy": accuracy
        }

        print(
            f"Precision@{k}: "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Accuracy@{k}: "
            f"{accuracy * 100:.2f}%"
        )

    # ========================================================
    # Combined results
    # ========================================================

    return {
        "knn": {
            "k": KNN_K,
            "accuracy": knn_accuracy,
            "train_samples": KNN_TRAIN_SAMPLES,
            "test_samples": KNN_TEST_SAMPLES,
        },

        "geometry": geometry,

        "retrieval": {
            "database_samples": RETRIEVAL_DATABASE_SAMPLES,
            "queries": RETRIEVAL_QUERIES,
            "results": retrieval_results,
        },
    }


# ============================================================
# SINGLE EXPERIMENT RUNNER
# ============================================================

def run_single_experiment(
    name,
    temperature,
    augmentation_strength,
    use_projection,
    train_loader,
    test_loader,
    train_indices,
    test_indices,
):
    """
    Train one condition and immediately evaluate its
    learned encoder representation.
    """

    print()
    print("=" * 70)
    print(f"RUNNING: {name}")
    print("=" * 70)

    print()
    print("Configuration:")
    print(
        f"  Temperature:       {temperature}"
    )
    print(
        f"  Augmentation:      {augmentation_strength}"
    )
    print(
        f"  Projection head:   {use_projection}"
    )
    print(
        f"  Epochs:             {EXPERIMENT_EPOCHS}"
    )
    print(
        f"  Batch size:         {EXPERIMENT_BATCH_SIZE}"
    )
    print(
        f"  Seed:               {SEED}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model, history, device = train_model(
        temperature=temperature,

        augmentation_strength=augmentation_strength,

        use_projection=use_projection,

        epochs=EXPERIMENT_EPOCHS,

        batch_size=EXPERIMENT_BATCH_SIZE,

        num_workers=EXPERIMENT_NUM_WORKERS,

        learning_rate=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY,

        representation_dim=REPRESENTATION_DIM,

        projection_hidden_dim=PROJECTION_HIDDEN_DIM,

        projection_dim=PROJECTION_DIM,

        seed=SEED,

        save_checkpoints=False,

        verbose=True
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        CHECKPOINT_DIR,
        f"{name}.pt"
    )

    config = {
        "temperature": temperature,
        "augmentation_strength": augmentation_strength,
        "use_projection": use_projection,
        "representation_dim": REPRESENTATION_DIM,
        "projection_hidden_dim": PROJECTION_HIDDEN_DIM,
        "projection_dim": PROJECTION_DIM,
    }

    save_experiment_checkpoint(
        model=model,
        history=history,
        config=config,
        path=checkpoint_path
    )

    print()
    print(
        f"Checkpoint saved to: "
        f"{checkpoint_path}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model_representation(
        model=model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        train_indices=train_indices,
        test_indices=test_indices,
    )

    # --------------------------------------------------------
    # Combine training + evaluation results
    # --------------------------------------------------------

    result = {
        "experiment": name,

        "config": config,

        "training": {
            "epochs": EXPERIMENT_EPOCHS,
            "batch_size": EXPERIMENT_BATCH_SIZE,
            "num_workers": EXPERIMENT_NUM_WORKERS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "history": history,
            "final_loss": history[-1]["loss"],
            "best_loss": min(
                item["loss"]
                for item in history
            ),
        },

        "evaluation": metrics,

        "checkpoint": checkpoint_path,
    }

    return result


# ============================================================
# EXPERIMENT 1: TEMPERATURE
# ============================================================

def run_temperature_experiment(
    train_loader,
    test_loader,
    train_indices,
    test_indices,
):
    print_experiment_header(
        "EXPERIMENT 1: TEMPERATURE"
    )

    temperatures = [
        0.1,
        0.5,
        1.0
    ]

    results = {}

    for temperature in temperatures:

        name = f"temperature_{temperature}"

        results[name] = run_single_experiment(
            name=name,
            temperature=temperature,
            augmentation_strength="strong",
            use_projection=True,
            train_loader=train_loader,
            test_loader=test_loader,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    output_path = os.path.join(
        RESULTS_DIR,
        "temperature_experiment.json"
    )

    save_json(
        results,
        output_path
    )

    print()
    print(
        f"Temperature experiment saved to: "
        f"{output_path}"
    )

    return results


# ============================================================
# EXPERIMENT 2: AUGMENTATION
# ============================================================

def run_augmentation_experiment(
    train_loader,
    test_loader,
    train_indices,
    test_indices,
):
    print_experiment_header(
        "EXPERIMENT 2: AUGMENTATION STRENGTH"
    )

    augmentation_strengths = [
        "weak",
        "strong"
    ]

    results = {}

    for augmentation_strength in augmentation_strengths:

        name = (
            f"augmentation_"
            f"{augmentation_strength}"
        )

        results[name] = run_single_experiment(
            name=name,
            temperature=0.5,
            augmentation_strength=augmentation_strength,
            use_projection=True,
            train_loader=train_loader,
            test_loader=test_loader,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    output_path = os.path.join(
        RESULTS_DIR,
        "augmentation_experiment.json"
    )

    save_json(
        results,
        output_path
    )

    print()
    print(
        f"Augmentation experiment saved to: "
        f"{output_path}"
    )

    return results


# ============================================================
# EXPERIMENT 3: PROJECTION HEAD
# ============================================================

def run_projection_experiment(
    train_loader,
    test_loader,
    train_indices,
    test_indices,
):
    print_experiment_header(
        "EXPERIMENT 3: PROJECTION HEAD"
    )

    projection_settings = [
        True,
        False
    ]

    results = {}

    for use_projection in projection_settings:

        name = (
            "with_projection"
            if use_projection
            else "without_projection"
        )

        results[name] = run_single_experiment(
            name=name,
            temperature=0.5,
            augmentation_strength="strong",
            use_projection=use_projection,
            train_loader=train_loader,
            test_loader=test_loader,
            train_indices=train_indices,
            test_indices=test_indices,
        )

    output_path = os.path.join(
        RESULTS_DIR,
        "projection_experiment.json"
    )

    save_json(
        results,
        output_path
    )

    print()
    print(
        f"Projection experiment saved to: "
        f"{output_path}"
    )

    return results


# ============================================================
# SUMMARY
# ============================================================

def print_metric_row(
    name,
    result
):
    evaluation = result["evaluation"]

    knn = (
        evaluation["knn"]["accuracy"]
        * 100
    )

    geometry = evaluation["geometry"]

    gap = geometry["similarity_gap"]

    precision_1 = (
        evaluation["retrieval"]
        ["results"]["1"]["precision"]
        * 100
    )

    precision_5 = (
        evaluation["retrieval"]
        ["results"]["5"]["precision"]
        * 100
    )

    loss = result["training"]["final_loss"]

    print(
        f"{name:25s}"
        f"{loss:>10.4f}"
        f"{knn:>12.2f}%"
        f"{gap:>12.4f}"
        f"{precision_1:>12.2f}%"
        f"{precision_5:>12.2f}%"
    )


def print_summary(
    temperature_results,
    augmentation_results,
    projection_results
):
    print()
    print("=" * 100)
    print("CONTROLLED EXPERIMENT SUMMARY")
    print("=" * 100)

    print()
    print(
        f"{'Condition':25s}"
        f"{'Loss':>10s}"
        f"{'kNN':>12s}"
        f"{'Gap':>12s}"
        f"{'P@1':>12s}"
        f"{'P@5':>12s}"
    )

    print("-" * 100)

    # --------------------------------------------------------
    # Temperature
    # --------------------------------------------------------

    print()
    print("TEMPERATURE")

    for name, result in temperature_results.items():
        print_metric_row(
            name,
            result
        )

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    print()
    print("AUGMENTATION")

    for name, result in augmentation_results.items():
        print_metric_row(
            name,
            result
        )

    # --------------------------------------------------------
    # Projection
    # --------------------------------------------------------

    print()
    print("PROJECTION")

    for name, result in projection_results.items():
        print_metric_row(
            name,
            result
        )

    print()
    print("=" * 100)


# ============================================================
# SAVE COMBINED RESULTS
# ============================================================

def save_combined_results(
    temperature_results,
    augmentation_results,
    projection_results,
):
    combined = {
        "experimental_design": {
            "seed": SEED,
            "epochs": EXPERIMENT_EPOCHS,
            "batch_size": EXPERIMENT_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,

            "evaluation": {
                "knn_train_samples": KNN_TRAIN_SAMPLES,
                "knn_test_samples": KNN_TEST_SAMPLES,
                "geometry_samples": GEOMETRY_SAMPLES,
                "retrieval_database_samples":
                    RETRIEVAL_DATABASE_SAMPLES,
                "retrieval_queries":
                    RETRIEVAL_QUERIES,
            }
        },

        "temperature": temperature_results,

        "augmentation": augmentation_results,

        "projection": projection_results,
    }

    path = os.path.join(
        RESULTS_DIR,
        "all_controlled_experiments.json"
    )

    save_json(
        combined,
        path
    )

    print()
    print(
        f"Combined results saved to: {path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "CONTROLLED CONTRASTIVE LEARNING EXPERIMENTS"
    )
    print("=" * 70)

    print()
    print("Experimental design:")
    print(
        "Temperature:       0.1, 0.5, 1.0"
    )
    print(
        "Augmentation:      weak, strong"
    )
    print(
        "Projection head:   with, without"
    )

    print()
    print(
        f"Epochs per run:    "
        f"{EXPERIMENT_EPOCHS}"
    )

    print(
        f"Batch size:        "
        f"{EXPERIMENT_BATCH_SIZE}"
    )

    print(
        f"Random seed:       "
        f"{SEED}"
    )

    print()
    print("Evaluation budget:")
    print(
        f"  k-NN train samples: "
        f"{KNN_TRAIN_SAMPLES}"
    )
    print(
        f"  k-NN test samples:  "
        f"{KNN_TEST_SAMPLES}"
    )
    print(
        f"  Geometry samples:   "
        f"{GEOMETRY_SAMPLES}"
    )
    print(
        f"  Retrieval database: "
        f"{RETRIEVAL_DATABASE_SAMPLES}"
    )
    print(
        f"  Retrieval queries:  "
        f"{RETRIEVAL_QUERIES}"
    )

    # --------------------------------------------------------
    # Load evaluation datasets once.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING EVALUATION DATA")
    print("=" * 70)

    train_loader, test_loader = get_dataloaders()

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # --------------------------------------------------------
    # Create fixed indices once.
    # --------------------------------------------------------

    train_indices, test_indices = create_fixed_indices(
        num_train=KNN_TRAIN_SAMPLES,
        num_test=KNN_TEST_SAMPLES,
    )

    print()
    print(
        "Fixed evaluation subsets created."
    )

    # --------------------------------------------------------
    # Experiment 1
    # --------------------------------------------------------

    temperature_results = (
        run_temperature_experiment(
            train_loader=train_loader,
            test_loader=test_loader,
            train_indices=train_indices,
            test_indices=test_indices,
        )
    )

    # --------------------------------------------------------
    # Experiment 2
    # --------------------------------------------------------

    augmentation_results = (
        run_augmentation_experiment(
            train_loader=train_loader,
            test_loader=test_loader,
            train_indices=train_indices,
            test_indices=test_indices,
        )
    )

    # --------------------------------------------------------
    # Experiment 3
    # --------------------------------------------------------

    projection_results = (
        run_projection_experiment(
            train_loader=train_loader,
            test_loader=test_loader,
            train_indices=train_indices,
            test_indices=test_indices,
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        temperature_results,
        augmentation_results,
        projection_results
    )

    # --------------------------------------------------------
    # Combined JSON
    # --------------------------------------------------------

    save_combined_results(
        temperature_results,
        augmentation_results,
        projection_results,
    )

    print()
    print("=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
