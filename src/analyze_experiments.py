import json
from pathlib import Path


RESULTS_DIR = Path("results/experiments")
OUTPUT_FILE = RESULTS_DIR / "experimental_analysis.txt"


def load(filename):
    with open(RESULTS_DIR / filename, "r") as f:
        return json.load(f)


def pct(value):
    return f"{value * 100:.2f}%"


def main():
    temperature = load("temperature_experiment.json")
    augmentation = load("augmentation_experiment.json")
    projection = load("projection_experiment.json")

    lines = []

    lines.append("=" * 80)
    lines.append("CONTROLLED EXPERIMENT ANALYSIS")
    lines.append("=" * 80)
    lines.append("")

    # ---------------------------------------------------------
    # Temperature
    # ---------------------------------------------------------

    lines.append("1. TEMPERATURE EXPERIMENT")
    lines.append("-" * 80)
    lines.append(
        "Controlled factor: InfoNCE temperature. "
        "All other training settings were held constant."
    )
    lines.append("")

    lines.append(
        f"{'Condition':<20}"
        f"{'Loss':>12}"
        f"{'kNN':>12}"
        f"{'P@1':>12}"
        f"{'P@5':>12}"
        f"{'Gap':>12}"
    )
    lines.append("-" * 80)

    for item in temperature.values():
        lines.append(
            f"{item['experiment']:<20}"
            f"{item['training']['final_loss']:>12.4f}"
            f"{pct(item['evaluation']['knn']['accuracy']):>12}"
            f"{pct(item['evaluation']['retrieval']['results']['1']['precision']):>12}"
            f"{pct(item['evaluation']['retrieval']['results']['5']['precision']):>12}"
            f"{item['evaluation']['geometry']['similarity_gap']:>12.4f}"
        )

    lines.append("")

    # ---------------------------------------------------------
    # Augmentation
    # ---------------------------------------------------------

    lines.append("2. AUGMENTATION EXPERIMENT")
    lines.append("-" * 80)
    lines.append(
        "Controlled factor: augmentation strength. "
        "All other training settings were held constant."
    )
    lines.append("")

    lines.append(
        f"{'Condition':<20}"
        f"{'Loss':>12}"
        f"{'kNN':>12}"
        f"{'P@1':>12}"
        f"{'P@5':>12}"
        f"{'Gap':>12}"
    )
    lines.append("-" * 80)

    for item in augmentation.values():
        lines.append(
            f"{item['experiment']:<20}"
            f"{item['training']['final_loss']:>12.4f}"
            f"{pct(item['evaluation']['knn']['accuracy']):>12}"
            f"{pct(item['evaluation']['retrieval']['results']['1']['precision']):>12}"
            f"{pct(item['evaluation']['retrieval']['results']['5']['precision']):>12}"
            f"{item['evaluation']['geometry']['similarity_gap']:>12.4f}"
        )

    lines.append("")

    # ---------------------------------------------------------
    # Projection head
    # ---------------------------------------------------------

    lines.append("3. PROJECTION HEAD EXPERIMENT")
    lines.append("-" * 80)
    lines.append(
        "Controlled factor: presence of the projection head. "
        "All other training settings were held constant."
    )
    lines.append("")

    lines.append(
        f"{'Condition':<20}"
        f"{'Loss':>12}"
        f"{'kNN':>12}"
        f"{'P@1':>12}"
        f"{'P@5':>12}"
        f"{'Gap':>12}"
    )
    lines.append("-" * 80)

    for item in projection.values():
        lines.append(
            f"{item['experiment']:<20}"
            f"{item['training']['final_loss']:>12.4f}"
            f"{pct(item['evaluation']['knn']['accuracy']):>12}"
            f"{pct(item['evaluation']['retrieval']['results']['1']['precision']):>12}"
            f"{pct(item['evaluation']['retrieval']['results']['5']['precision']):>12}"
            f"{item['evaluation']['geometry']['similarity_gap']:>12.4f}"
        )

    lines.append("")

    # ---------------------------------------------------------
    # Detailed observations
    # ---------------------------------------------------------

    lines.append("4. KEY OBSERVATIONS")
    lines.append("-" * 80)

    # Temperature
    t01 = temperature["temperature_0.1"]
    t05 = temperature["temperature_0.5"]
    t10 = temperature["temperature_1.0"]

    lines.append("")
    lines.append("Temperature:")
    lines.append(
        f"- tau=0.1 produced the lowest final loss "
        f"({t01['training']['final_loss']:.4f})."
    )
    lines.append(
        f"- tau=0.1 achieved kNN accuracy of "
        f"{pct(t01['evaluation']['knn']['accuracy'])}."
    )
    lines.append(
        f"- tau=0.5 achieved the largest similarity gap "
        f"({t05['evaluation']['geometry']['similarity_gap']:.4f})."
    )
    lines.append(
        f"- tau=1.0 had the highest final loss "
        f"({t10['training']['final_loss']:.4f}) and the lowest kNN accuracy "
        f"({pct(t10['evaluation']['knn']['accuracy'])})."
    )
    lines.append(
        "- Loss values across temperatures should not be interpreted "
        "as directly comparable representation-quality scores because "
        "temperature changes the scale/sharpness of the InfoNCE logits."
    )

    # Augmentation
    weak = augmentation["augmentation_weak"]
    strong = augmentation["augmentation_strong"]

    lines.append("")
    lines.append("Augmentation:")
    lines.append(
        f"- Weak augmentation achieved kNN accuracy of "
        f"{pct(weak['evaluation']['knn']['accuracy'])}."
    )
    lines.append(
        f"- Strong augmentation achieved kNN accuracy of "
        f"{pct(strong['evaluation']['knn']['accuracy'])}."
    )
    lines.append(
        f"- Strong augmentation increased the similarity gap from "
        f"{weak['evaluation']['geometry']['similarity_gap']:.4f} "
        f"to {strong['evaluation']['geometry']['similarity_gap']:.4f}."
    )
    lines.append(
        f"- Strong augmentation also improved Precision@1 from "
        f"{pct(weak['evaluation']['retrieval']['results']['1']['precision'])} "
        f"to {pct(strong['evaluation']['retrieval']['results']['1']['precision'])}."
    )
    lines.append(
        "- Strong augmentation produced a higher training loss, showing "
        "that lower contrastive loss does not necessarily correspond "
        "to better downstream representation quality."
    )

    # Projection
    with_projection = projection["with_projection"]
    without_projection = projection["without_projection"]

    lines.append("")
    lines.append("Projection head:")
    lines.append(
        f"- With projection: kNN="
        f"{pct(with_projection['evaluation']['knn']['accuracy'])}, "
        f"P@1="
        f"{pct(with_projection['evaluation']['retrieval']['results']['1']['precision'])}, "
        f"gap="
        f"{with_projection['evaluation']['geometry']['similarity_gap']:.4f}."
    )
    lines.append(
        f"- Without projection: kNN="
        f"{pct(without_projection['evaluation']['knn']['accuracy'])}, "
        f"P@1="
        f"{pct(without_projection['evaluation']['retrieval']['results']['1']['precision'])}, "
        f"gap="
        f"{without_projection['evaluation']['geometry']['similarity_gap']:.4f}."
    )
    lines.append(
        "- The no-projection configuration performed better on the "
        "measured downstream metrics under this two-epoch training budget."
    )
    lines.append(
        "- The final losses were very similar, so the difference cannot "
        "be explained simply by optimization loss."
    )
    lines.append(
        "- This is an unexpected result relative to the intended role of "
        "a projection head and should be treated as an experiment-specific "
        "finding rather than a general conclusion that projection heads "
        "are harmful."
    )

    # ---------------------------------------------------------
    # Failure analysis
    # ---------------------------------------------------------

    lines.append("")
    lines.append("5. FAILURE / UNEXPECTED-RESULT ANALYSIS")
    lines.append("-" * 80)

    lines.append("")
    lines.append("Case A: Strong augmentation produced higher loss but better representations")
    lines.append("")
    lines.append("What happened:")
    lines.append(
        f"Strong augmentation had a final loss of "
        f"{strong['training']['final_loss']:.4f}, compared with "
        f"{weak['training']['final_loss']:.4f} for weak augmentation. "
        f"However, strong augmentation achieved "
        f"{pct(strong['evaluation']['knn']['accuracy'])} kNN accuracy "
        f"versus {pct(weak['evaluation']['knn']['accuracy'])}."
    )
    lines.append("")
    lines.append("Why:")
    lines.append(
        "Strong augmentation makes the positive-pair matching task more "
        "difficult because the two views can differ substantially. "
        "The model therefore has to learn features that remain stable "
        "under these transformations. A higher loss can therefore "
        "coexist with a more useful representation."
    )
    lines.append("")
    lines.append("Possible improvement:")
    lines.append(
        "Train for more epochs and compare several augmentation strengths "
        "using the same downstream metrics. This would help determine "
        "whether the observed advantage remains after longer optimization."
    )

    lines.append("")
    lines.append("Case B: Removing the projection head performed better")
    lines.append("")
    lines.append("What happened:")
    lines.append(
        f"The no-projection model achieved "
        f"{pct(without_projection['evaluation']['knn']['accuracy'])} "
        f"kNN accuracy and a similarity gap of "
        f"{without_projection['evaluation']['geometry']['similarity_gap']:.4f}, "
        f"compared with "
        f"{pct(with_projection['evaluation']['knn']['accuracy'])} and "
        f"{with_projection['evaluation']['geometry']['similarity_gap']:.4f} "
        f"with the projection head."
    )
    lines.append("")
    lines.append("Why:")
    lines.append(
        "The experiment was limited to two training epochs, so the "
        "projection head may not have had sufficient optimization time. "
        "In addition, downstream evaluation uses the encoder representation "
        "h rather than the projected representation z. The projection "
        "head can therefore alter the optimization objective without "
        "necessarily improving the representation being evaluated."
    )
    lines.append("")
    lines.append("Possible improvement:")
    lines.append(
        "Repeat the comparison with longer training and explicitly "
        "evaluate both encoder representations h and projected "
        "representations z. Multiple random seeds would also help "
        "determine whether the observed difference is robust."
    )

    lines.append("")
    lines.append("Case C: Random encoder showed extremely high similarity")
    lines.append("")
    lines.append("What happened:")
    lines.append(
        "The random encoder produced same-class cosine similarity of "
        "0.978887 and different-class similarity of 0.976032, resulting "
        "in a similarity gap of only 0.002855."
    )
    lines.append("")
    lines.append("Why:")
    lines.append(
        "The random encoder's embeddings are highly concentrated in a "
        "similar direction. High absolute cosine similarity therefore "
        "does not imply useful semantic structure. The very small "
        "same-vs-different gap indicates that the representation does "
        "not meaningfully distinguish classes."
    )
    lines.append("")
    lines.append("Possible improvement:")
    lines.append(
        "Evaluate both absolute similarity and relative class separation. "
        "Embedding variance, similarity gaps, kNN accuracy, and retrieval "
        "metrics together provide a more reliable representation-quality "
        "assessment than raw similarity alone."
    )

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF ANALYSIS")
    lines.append("=" * 80)

    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines))

    print(f"Analysis written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
