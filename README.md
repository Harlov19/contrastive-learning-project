# Contrastive Learning on CIFAR-10

A complete contrastive-learning pipeline for learning and evaluating visual representations on CIFAR-10.

The central research question is:

> **How do different training and model design choices affect the quality and structure of learned representations?**

The project implements a two-view contrastive-learning pipeline and evaluates the resulting representation using classification, embedding geometry, visualization, and nearest-neighbor retrieval.

---

## 1. Project Overview

The pipeline follows:

```text
CIFAR-10 image
      │
      ├──────────────┐
      ↓              ↓
 Augmentation 1   Augmentation 2
      ↓              ↓
    View 1         View 2
      │              │
      └──────┬───────┘
             ↓
          Encoder
             ↓
       Representation h
             ↓
       Projection Head
             ↓
       Normalized z
             ↓
        InfoNCE Loss
             ↓
     Learned Representation
```

The two augmented views of the same image form a positive pair. Other samples in the batch provide in-batch negative examples.

---

## 2. Objectives

The implementation demonstrates:

- Positive and negative pairs
- Data augmentation and invariance
- CNN encoder
- Projection head
- Embedding normalization
- Cosine similarity
- InfoNCE / NT-Xent loss
- Temperature scaling
- Representation geometry
- k-nearest-neighbor classification
- PCA visualization
- Nearest-neighbor retrieval
- Controlled experimentation
- Failure and unexpected-result analysis

---

## 3. Dataset

The project uses **CIFAR-10**.

The dataset contains:

- 50,000 training images
- 10,000 test images
- 10 semantic classes
- RGB images of size `32 × 32`

The classes are:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

Images are normalized using the standard CIFAR-10 channel statistics:

```text
mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)
```

---

## 4. Data Augmentation

Two augmentation configurations are implemented.

### Weak augmentation

- Random horizontal flip
- Tensor conversion
- CIFAR-10 normalization

### Strong augmentation

- Random resized crop
- Random horizontal flip
- Color jitter
- Random grayscale
- Tensor conversion
- CIFAR-10 normalization

For every training image, two independently sampled augmented views are generated:

```text
x
├── t₁(x) → view 1
└── t₂(x) → view 2
```

These views constitute a positive pair.

The purpose is to encourage the encoder to learn representations that remain useful under image transformations while separating different samples.

---

## 5. Model Architecture

### Encoder

The encoder is a convolutional neural network consisting of four convolutional blocks:

```text
Input: 3 × 32 × 32

Conv 3 → 64
BatchNorm
ReLU

Conv 64 → 128, stride 2
BatchNorm
ReLU

Conv 128 → 256, stride 2
BatchNorm
ReLU

Conv 256 → 512, stride 2
BatchNorm
ReLU

Adaptive Average Pooling
Linear → 128
```

The encoder produces a **128-dimensional representation**:

```text
h ∈ R¹²⁸
```

### Projection head

The default model uses:

```text
128
 ↓
Linear
 ↓
ReLU
 ↓
128
 ↓
Linear
 ↓
64
 ↓
L2 normalization
```

The projection head produces a 64-dimensional normalized vector:

```text
z ∈ R⁶⁴
```

The encoder representation `h` is used for downstream representation evaluation, while `z` is used by the contrastive objective.

The implementation also supports training without the projection head for controlled experimentation.

---

## 6. Contrastive Objective

The implementation uses an NT-Xent / InfoNCE-style loss.

For an anchor representation and its positive counterpart, the objective encourages:

```text
similarity(anchor, positive) ↑
similarity(anchor, negative) ↓
```

Other samples in the batch act as in-batch negatives.

Before calculating the similarity matrix, representations are L2-normalized:

```text
ẑ = z / ||z||
```

Therefore, the dot product between normalized embeddings corresponds to cosine similarity.

The temperature parameter controls how strongly similarity differences affect the softmax distribution.

---

## 7. Training Configuration

Main training configuration:

| Parameter                   |    Value |
| --------------------------- | -------: |
| Dataset                     | CIFAR-10 |
| Epochs                      |        3 |
| Batch size                  |      128 |
| Learning rate               | 3 × 10⁻⁴ |
| Weight decay                | 1 × 10⁻⁴ |
| Temperature                 |      0.5 |
| Encoder dimension           |      128 |
| Projection hidden dimension |      128 |
| Projection dimension        |       64 |
| Random seed                 |       42 |
| Device                      |      CPU |

Main training loss:

```text
Epoch 1: 4.2446
Epoch 2: 4.0331
Epoch 3: 3.9762
```

The decreasing loss indicates that the contrastive objective was being optimized during training.

---

## 8. Representation Evaluation

The learned encoder representation is evaluated using several complementary methods.

### 8.1 k-NN classification

A 5-nearest-neighbor classifier is applied to normalized encoder embeddings.

Results:

| Representation       | k-NN Accuracy |
| -------------------- | ------------: |
| Random encoder       |        33.19% |
| Trained encoder      |        48.76% |
| Absolute improvement |     +16.11 pp |

The trained representation therefore contains substantially more class-relevant information than the randomly initialized encoder under this evaluation.

### 8.2 Representation geometry

The average cosine similarity is measured separately for same-class and different-class pairs.

| Representation  | Same-class | Different-class |    Gap |
| --------------- | ---------: | --------------: | -----: |
| Random encoder  |     0.9789 |          0.9760 | 0.0029 |
| Trained encoder |     0.5217 |          0.3632 | 0.1584 |

The random encoder produces extremely similar embeddings for many samples, but the same-class/different-class difference is almost zero.

After contrastive training, the overall similarity decreases while the class-dependent gap increases substantially.

This is more informative than looking at raw similarity alone.

### 8.3 PCA

PCA is used to project the 128-dimensional representation into two dimensions.

PCA is treated primarily as a visualization tool. Explained variance is **not** interpreted as a direct measure of representation quality.

### 8.4 Retrieval

Nearest-neighbor retrieval is performed using cosine similarity.

Results over 1,000 queries:

| Metric      | Random | Trained |
| ----------- | -----: | ------: |
| Precision@1 | 29.10% |  44.60% |
| Precision@5 | 25.22% |  38.46% |
| Accuracy@1  | 29.10% |  44.60% |
| Accuracy@5  | 68.60% |  78.30% |

The trained representation therefore retrieves semantically matching CIFAR-10 examples more frequently than the random baseline.

---

## 9. Controlled Experiments

Three controlled experiments were performed. Each experiment changes one major factor while keeping the other principal training settings fixed.

All controlled experiments use:

- 2 epochs
- batch size 128
- learning rate 3 × 10⁻⁴
- weight decay 1 × 10⁻⁴
- seed 42
- CIFAR-10 training set
- identical evaluation procedure

### Experiment 1 — Temperature

| Temperature | Final Loss |   k-NN | Similarity Gap |    P@1 |    P@5 |
| ----------: | ---------: | -----: | -------------: | -----: | -----: |
|         0.1 |     0.8833 | 46.85% |         0.0965 | 43.50% | 39.06% |
|         0.5 |     4.0331 | 41.20% |         0.1244 | 39.80% | 36.14% |
|         1.0 |     4.7397 | 37.90% |         0.1113 | 37.30% | 33.98% |

The lowest loss occurred at temperature 0.1, and it also produced the strongest k-NN and retrieval metrics in this experiment.

However, the loss values should not be compared in isolation across temperatures because temperature directly rescales the logits and therefore changes the scale and sharpness of the optimization objective.

The experiment demonstrates that temperature affects both optimization and embedding geometry.

### Experiment 2 — Augmentation

| Augmentation | Final Loss |   k-NN | Similarity Gap |    P@1 |    P@5 |
| ------------ | ---------: | -----: | -------------: | -----: | -----: |
| Weak         |     3.6245 | 32.05% |         0.0584 | 27.10% | 25.48% |
| Strong       |     4.0331 | 41.20% |         0.1244 | 39.80% | 36.14% |

Strong augmentation produced a higher contrastive loss but substantially better downstream representation measurements.

This illustrates an important distinction:

> **Lower contrastive loss does not automatically imply a better downstream representation.**

The stronger transformations make the positive-pair matching problem harder, but they also encourage the encoder to become invariant to larger changes in appearance.

### Experiment 3 — Projection Head

| Configuration      | Final Loss |   k-NN | Similarity Gap |    P@1 |    P@5 |
| ------------------ | ---------: | -----: | -------------: | -----: | -----: |
| With projection    |     4.0331 | 41.20% |         0.1244 | 39.80% | 36.14% |
| Without projection |     4.0233 | 45.20% |         0.1583 | 41.90% | 37.86% |

Under the short two-epoch experimental budget, removing the projection head produced stronger downstream encoder metrics.

This is treated as an **unexpected result**, rather than evidence that projection heads are generally harmful. The projection-head configuration may require a longer training schedule or a different optimization regime to show its intended benefit.

---

## 10. Failure and Unexpected-Result Analysis

### Case 1 — Strong augmentation had higher loss but better representations

**What happened**

Strong augmentation produced a final loss of 4.0331 compared with 3.6245 for weak augmentation, yet k-NN accuracy increased from 32.05% to 41.20%.

**Why**

The strong augmentation makes the two views of an image more different, increasing the difficulty of the contrastive matching task. At the same time, solving this harder task encourages invariance to transformations that preserve semantic content.

**Possible improvement**

Evaluate additional augmentation strengths and train for more epochs to determine whether the observed improvement persists.

---

### Case 2 — Removing the projection head performed better

**What happened**

Without the projection head, k-NN accuracy increased from 41.20% to 45.20% and the similarity gap increased from 0.1244 to 0.1583.

**Why**

The experiment was limited to two epochs. The additional projection layer changes the optimization pathway between the contrastive objective and the encoder. Under this short CPU training budget, the encoder without the additional head produced better measured downstream representations.

**Possible improvement**

Train both configurations for substantially longer, tune their optimization settings independently, and evaluate both the encoder representation `h` and projection representation `z`.

---

### Case 3 — Random embeddings had extremely high similarity

**What happened**

The random encoder had same-class cosine similarity of 0.9789 and different-class similarity of 0.9760.

**Why**

The randomly initialized convolutional network maps many inputs into a highly concentrated region of the representation space. High overall similarity therefore does not mean the representation is useful.

**Possible improvement**

Monitor embedding variance, average pairwise similarity, and same-vs-different similarity simultaneously rather than using raw similarity as the only geometry metric.

---

## 11. Main Findings

The experiments show that contrastive learning changed the representation in several measurable ways.

First, downstream k-NN performance increased from 33.19% for the random encoder to 48.76% for the trained encoder.

Second, the same-class/different-class similarity gap increased from approximately 0.0029 to 0.1584.

Third, retrieval performance improved substantially, with Precision@1 increasing from 29.10% to 44.60% and Accuracy@5 increasing from 68.60% to 78.30%.

The controlled experiments additionally demonstrate that:

- Temperature changes the optimization landscape and embedding geometry.
- Strong augmentation can produce better representations even when its contrastive loss is higher.
- Projection-head behavior depends on the training regime and should not be judged from loss alone.
- Representation quality requires multiple measurements rather than a single metric.

---

## 12. Limitations

The main limitations are:

1. The main model was trained for only three epochs.
2. Controlled experiments used only two epochs per configuration because training was performed on CPU.
3. Only one CNN architecture was investigated.
4. Only a limited set of temperatures and augmentation configurations was tested.
5. Retrieval evaluation used a fixed test-database/query setup.
6. PCA provides visualization but is not a direct measure of representation quality.
7. The experiments do not establish that the observed configuration is globally optimal; they characterize behavior under the specified experimental budget.

These limitations are important when interpreting the experimental conclusions.

---

## 13. Reproducibility

Create the environment and install the required packages, then run the components from the repository root.

Main training:

```bash
python src/train.py
```

Representation evaluation:

```bash
python src/evaluate.py
```

Retrieval evaluation:

```bash
python src/retrieval.py
```

Controlled experiments:

```bash
python src/experiments.py
```

Experiment visualizations:

```bash
python src/plot_experiments.py
```

Experiment analysis:

```bash
python src/analyze_experiments.py
```

Results are written to:

```text
results/
├── training_history.json
├── representation_evaluation.txt
├── knn_results.txt
├── retrieval_results.txt
├── pca_results.txt
└── experiments/
    ├── all_controlled_experiments.json
    ├── experimental_analysis.txt
    └── figures/
```

Model checkpoints are stored under:

```text
checkpoints/
```

---

## 14. Repository Structure

```text
contrastive-learning-project/
│
├── data/
│
├── src/
│   ├── augmentations.py
│   ├── dataset.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── evaluate.py
│   ├── retrieval.py
│   ├── visualization.py
│   ├── experiments.py
│   ├── plot_experiments.py
│   └── analyze_experiments.py
│
├── experiments/
├── results/
├── checkpoints/
├── notebooks/
├── report/
└── README.md
```

Large generated artifacts such as datasets, model checkpoints, and experiment outputs are excluded from Git tracking through `.gitignore`.

---

## 15. References

- Chen et al., _A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)_.
- Wang & Isola, _Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere_.
- Krizhevsky, _CIFAR-10 Dataset_.
