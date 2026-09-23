import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_DIR = Path("results/experiments")
OUTPUT_DIR = RESULTS_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results(filename):
    path = RESULTS_DIR / filename

    with open(path, "r") as f:
        return json.load(f)


def ordered_results(data):
    """
    Convert the dictionary of experiment results into an ordered list.
    Python preserves insertion order, but sorting by experiment name
    makes the intended experimental ordering explicit.
    """
    return list(data.values())


def plot_metric(
    labels,
    values,
    title,
    ylabel,
    filename,
    percentage=False,
):
    plt.figure(figsize=(8, 5))

    x = list(range(len(labels)))

    if percentage:
        values = [value * 100 for value in values]

    plt.plot(
        x,
        values,
        marker="o",
        linewidth=2,
    )

    plt.xticks(x, labels)
    plt.xlabel("Experimental Condition")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(True, alpha=0.3)

    for i, value in enumerate(values):
        plt.annotate(
            f"{value:.2f}",
            (x[i], value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def plot_experiment_group(data, group_name, filename_prefix):
    results = ordered_results(data)

    labels = [
        item["experiment"]
        for item in results
    ]

    # ---------------------------------------------------------
    # Final training loss
    # ---------------------------------------------------------

    losses = [
        item["training"]["final_loss"]
        for item in results
    ]

    plot_metric(
        labels=labels,
        values=losses,
        title=f"{group_name}: Final Training Loss",
        ylabel="Final InfoNCE Loss",
        filename=f"{filename_prefix}_loss.png",
    )

    # ---------------------------------------------------------
    # k-NN accuracy
    # ---------------------------------------------------------

    knn = [
        item["evaluation"]["knn"]["accuracy"]
        for item in results
    ]

    plot_metric(
        labels=labels,
        values=knn,
        title=f"{group_name}: k-NN Representation Accuracy",
        ylabel="k-NN Accuracy (%)",
        filename=f"{filename_prefix}_knn.png",
        percentage=True,
    )

    # ---------------------------------------------------------
    # Retrieval Precision@1
    # ---------------------------------------------------------

    precision_at_1 = [
        item["evaluation"]["retrieval"]["results"]["1"]["precision"]
        for item in results
    ]

    plot_metric(
        labels=labels,
        values=precision_at_1,
        title=f"{group_name}: Retrieval Precision@1",
        ylabel="Precision@1 (%)",
        filename=f"{filename_prefix}_precision_at_1.png",
        percentage=True,
    )

    # ---------------------------------------------------------
    # Retrieval Precision@5
    # ---------------------------------------------------------

    precision_at_5 = [
        item["evaluation"]["retrieval"]["results"]["5"]["precision"]
        for item in results
    ]

    plot_metric(
        labels=labels,
        values=precision_at_5,
        title=f"{group_name}: Retrieval Precision@5",
        ylabel="Precision@5 (%)",
        filename=f"{filename_prefix}_precision_at_5.png",
        percentage=True,
    )

    # ---------------------------------------------------------
    # Same-vs-different class similarity gap
    # ---------------------------------------------------------

    similarity_gap = [
        item["evaluation"]["geometry"]["similarity_gap"]
        for item in results
    ]

    plot_metric(
        labels=labels,
        values=similarity_gap,
        title=f"{group_name}: Representation Similarity Gap",
        ylabel="Same-Class − Different-Class Cosine Similarity",
        filename=f"{filename_prefix}_similarity_gap.png",
    )

    # ---------------------------------------------------------
    # Same-class similarity
    # ---------------------------------------------------------

    same_class = [
        item["evaluation"]["geometry"]["same_class_mean"]
        for item in results
    ]

    plot_metric(
        labels=labels,
        values=same_class,
        title=f"{group_name}: Same-Class Similarity",
        ylabel="Mean Cosine Similarity",
        filename=f"{filename_prefix}_same_class_similarity.png",
    )

    # ---------------------------------------------------------
    # Different-class similarity
    # ---------------------------------------------------------

    different_class = [
        item["evaluation"]["geometry"]["different_class_mean"]
        for item in results
    ]

    plot_metric(
        labels=labels,
        values=different_class,
        title=f"{group_name}: Different-Class Similarity",
        ylabel="Mean Cosine Similarity",
        filename=f"{filename_prefix}_different_class_similarity.png",
    )


def main():
    print("Generating controlled-experiment visualizations...")

    # ---------------------------------------------------------
    # Load experiment results
    # ---------------------------------------------------------

    temperature = load_results(
        "temperature_experiment.json"
    )

    augmentation = load_results(
        "augmentation_experiment.json"
    )

    projection = load_results(
        "projection_experiment.json"
    )

    # ---------------------------------------------------------
    # Generate temperature plots
    # ---------------------------------------------------------

    print("\nTemperature experiments:")

    plot_experiment_group(
        temperature,
        group_name="Temperature",
        filename_prefix="temperature",
    )

    # ---------------------------------------------------------
    # Generate augmentation plots
    # ---------------------------------------------------------

    print("Augmentation experiments:")

    plot_experiment_group(
        augmentation,
        group_name="Augmentation",
        filename_prefix="augmentation",
    )

    # ---------------------------------------------------------
    # Generate projection-head plots
    # ---------------------------------------------------------

    print("Projection-head experiments:")

    plot_experiment_group(
        projection,
        group_name="Projection Head",
        filename_prefix="projection",
    )

    # ---------------------------------------------------------
    # Show generated files
    # ---------------------------------------------------------

    print("\nGenerated figures:")

    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {path}")

    print("\nVisualization generation complete.")


if __name__ == "__main__":
    main()
