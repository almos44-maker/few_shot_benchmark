# Few-shot classification benchmark for foundation models

Computer Vision

Benchmarking three pre-trained visual backbones with three lightweight classification heads on two datasets, in the few-shot (C-way K-shot) setting.

## Overview
Given a frozen pre-trained backbone, we extract feature embeddings once, then fit lightweight classifiers on small support sets and evaluate on query sets.
Each result is the mean accuracy over 10,000 randomly sampled episodes (seed 42), for every combination of dataset × backbone × head × K.

- **Backbones (frozen):** DINOv2, CLIP RN50, ConvNeXt
- **Heads:** Prototypical, Ridge Regression, Linear Probing
- **Datasets:** EuroSAT (10-way satellite), StanfordCars (196-way fine-grained)
- **Shots:** K->{1, 2, 4, 16}, 15 query images per class
- **Task 5:** a custom adapter head that beats the baselines on most configurations

## Repository structure
1. **Feature extraction**- loads each backbone via HuggingFace `transformers` and the OpenAI `clip` library, runs every image through the frozen backbone once, and saves the
   embeddings to `saved_features/{dataset}_{backbone}.pt`.

2. **Benchmarking**- samples episodes from the saved features and fits the three heads (plus task 5's adapter), producing the results table, bar charts, scaling plots,
   training-dynamics curves, and UMAP feature-space visualizations.

## Setup
pip install torch torchvision transformers datasets scikit-learn umap-learn matplotlib seaborn pandas

pip install git+https://github.com/openai/CLIP.git

## How to run

1. Run the feature-extraction section once- this populates `saved_features/`.
Recommended on a GPU (Colab); it saves significant time on later runs.

2. Run the benchmarking section- it loads the saved features and reproduces all tablesand figures.
Results are cached to `few_shot_benchmark.csv`, so completed configurations are skipped on re-runs.



## Method notes

- **Prototypical:** each class is a mean feature vector; queries assigned by nearest (Euclidean) prototype.
- **Ridge:** closed-form W=((X^T)X + (lambda)I)^(-1) (X^T)Y
- **Linear probing:** single linear layer trained with cross-entropy (Adam, 20 epochs).
- **Adapter (Task 5):** a residual bottleneck adapter over the frozen features with a cosine classifier initialized from the class means
  (CLIP-Adapter [1] + prototypical init [10] + CLIP-style cosine head [9]). Trained per episode on the support set only- no label leakage, frozen backbone.

- **Normalization:** CLIP features are L2-normalized at extraction. DINOv2 and ConvNeXt are left in their native scale.


## Key results

- **Head comparison:** Prototypical/Ridge lead at low K (no fitting needed). Linear and Ridge lead at K=16 (enough data to fit boundaries).

- **Ridge instability:** Ridge collapses when the support count N=C*K is near the feature dimension D (StanfordCars/DINOv2 K=2 -> ~4%),
  a matrix-conditioning effect that moves with backbone dimension.

- **Domain shift:** all backbones handle EuroSAT (out-of-domain satellite) well because the task is coarse.
  StanfordCars (fine-grained, in-domain) is gated by each backbone's pre-training objective- CLIP best, ConvNeXt worst.

- **Adapter:** wins or ties on 15/24 configurations, with its clearest gains on StanfordCars/DINOv2 and at high K.

## References

[1] Gao et al., CLIP-Adapter (2024) · [9] Radford et al., CLIP (2021) ·

[10] Snell et al., Prototypical Networks (2017)
