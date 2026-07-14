# Low-Vision Perceptual Color Difference

This repository contains the experimental evaluation of perceptual color-difference prediction models developed for people with low vision.

The objective of this work is to predict how people with low vision perceive the difference between two colors represented in the CIELAB color space.

For each color pair, the models predict a probability distribution over three perceptual response categories:

1. No visible difference
2. Slight or just-noticeable difference
3. Clear difference

The repository contains three experiments designed to evaluate model performance under within-dataset, cross-illumination, and cross-dataset conditions.

## Research Background

Conventional color-difference formulas, such as CIELAB ΔE, are mainly based on visual observations obtained from people with normal vision.

However, people with low vision may perceive color differences differently because of factors such as:

- reduced visual acuity
- reduced contrast sensitivity
- impaired color discrimination
- ocular disease
- differences in visual-field condition
- individual variation in residual vision

As a result, a color pair considered clearly distinguishable under a conventional color-difference formula may not necessarily be perceived in the same way by a person with low vision.

This project evaluates whether data-driven models can better represent perceptual color-difference responses collected from observers with low vision.

## Repository Contents

The repository contains three executed notebooks:

- `notebooks/01_D1_Random_75_25.ipynb`
- `notebooks/02_D1_Leave_One_Illumination_Out.ipynb`
- `notebooks/03_Full_D1_to_D2.ipynb`

Each notebook includes:

- parameter configuration
- data preprocessing
- baseline definitions
- neural-network definitions
- model training
- test evaluation
- prediction export
- final result tables

## Datasets

Two experimental datasets are used: D1 and D2.

The datasets are not included in this repository.

They contain experimental responses collected from human observers and are therefore not redistributed here.

### D1 Dataset

D1 is the primary dataset used for model development and evaluation.

It contains:

- 200 unique color pairs
- four illumination conditions
- 800 total samples
- multiple observer responses for each sample

Each of the 200 color pairs was evaluated under four different illumination conditions:

    200 color pairs × 4 illumination conditions = 800 samples

Each sample contains two colors represented in CIELAB coordinates:

    Color 1: L1*, a1*, b1*
    Color 2: L2*, a2*, b2*

The input therefore consists of six numerical values:

    L1*, a1*, b1*, L2*, a2*, b2*

For each sample, observers with low vision evaluated the perceptual difference between the two colors.

The individual observer responses were converted into a probability distribution over three response categories:

- Category 0: no visible difference
- Category 1: slight or just-noticeable difference
- Category 2: clear difference

For example, a target distribution may have the following form:

    [0.10, 0.25, 0.65]

This means that:

- 10% of the observer responses indicated no visible difference
- 25% indicated a slight difference
- 65% indicated a clear difference

The models are trained to predict this complete response distribution rather than a single class label.

#### D1 Illumination Structure

D1 contains four illumination conditions, with 200 samples per condition.

The leave-one-illumination-out experiment uses the following structure:

| Fold | Training conditions | Test condition |
|---|---|---|
| 1 | Lights 2, 3, and 4 | Light 1 |
| 2 | Lights 1, 3, and 4 | Light 2 |
| 3 | Lights 1, 2, and 4 | Light 3 |
| 4 | Lights 1, 2, and 3 | Light 4 |

Each fold therefore contains:

- 600 training samples
- 200 test samples

The purpose of this experiment is to determine whether a model trained under three illumination conditions can generalize to an illumination condition that was not included during training.

### D2 Dataset

D2 is an independent dataset used for cross-dataset evaluation.

It contains:

- 102 color pairs
- six observer responses per color pair
- color pairs different from those in D1
- an observer group different from that used for D1

As in D1, each color pair is represented by two CIELAB triplets:

    L1*, a1*, b1*, L2*, a2*, b2*

The six observer responses for each color pair are converted into a three-category probability distribution.

D2 is not used for neural-network training.

In the cross-dataset experiment:

- all 800 D1 samples are used for training
- all 102 D2 samples are used for testing

This is the most challenging experiment because D1 and D2 differ in both color-pair composition and observer population.

## Data Availability

The original D1 and D2 datasets are not included in this repository.

To run the notebooks, local copies of the datasets are required.

The file paths must be updated in the parameter cell of each notebook.

For example:

```python
D1_PATH = "/path/to/D1.csv"
D2_PATH = "/path/to/D2.csv"
```

D2 is required only for the cross-dataset experiment.

The expected D1 structure includes:

- six CIELAB input columns
- observer-response columns or three precomputed probability columns
- 800 rows
- four illumination conditions

The expected D2 structure includes:

- six CIELAB input columns
- six observer-response columns or three precomputed probability columns
- 102 rows

## Evaluated Methods

Six methods are evaluated in each experiment:

1. Naive Predictor
2. ED Regressor
3. M1: Siamese
4. M2: Siamese+Lab
5. M3: Siamese+ΔE
6. M4: Siamese+ΔE+Lab

## Baseline Methods

### Naive Predictor

The Naive Predictor calculates the mean target distribution of the training set.

The same distribution is then used as the prediction for every test sample.

For example:

    mean training distribution = [0.15, 0.20, 0.65]

Every test sample receives the prediction:

    [0.15, 0.20, 0.65]

For the leave-one-illumination-out experiment, the Naive distribution is recalculated independently for each fold using only the corresponding three-light training subset.

### ED Regressor

The ED Regressor uses the original CIELAB color difference, ΔEab, as its independent variable.

For each color pair:

    ΔEab = sqrt(
        (L1* - L2*)²
        + (a1* - a2*)²
        + (b1* - b2*)²
    )

Three nonlinear functions are fitted independently, one for each perceptual response category:

    p_k(d) = a_k × exp(b_k / d)

where:

- `d` is the CIELAB ΔEab value
- `a_k` and `b_k` are fitted parameters
- `k` represents one of the three response categories

The three predicted values are clipped to positive values and normalized so that their sum is one.

The ED Regressor contains six fitted parameters in total.

## Neural-Network Models

All four neural models use a Siamese architecture.

The two input colors are processed independently by the same encoder with shared weights.

### Shared Siamese Encoder

Each input color contains three values:

    L*, a*, b*

The encoder architecture is:

    Linear: 3 → 32
    ReLU

    Linear: 32 → 64
    ReLU

    Linear: 64 → 64
    ReLU

    Linear: 64 → 64
    ReLU

    Linear: 64 → 64

The final encoder layer has no activation function.

The encoder produces a 64-dimensional representation for each input color.

The Siamese distance is calculated as the Euclidean distance between the two embeddings:

    d_siamese = ||embedding_1 - embedding_2||₂

### Prediction Network

All four neural models use the same prediction-network structure:

    Linear: input dimension → 32
    ReLU

    Linear: 32 → 32
    ReLU

    Linear: 32 → 16
    ReLU

    Linear: 16 → 8
    ReLU

    Linear: 8 → 3

The final three outputs are logits.

Softmax is applied during evaluation to obtain the predicted three-category probability distribution.

No dropout layers are used.

## Model-Specific Inputs

### M1: Siamese

M1 uses only the learned Siamese distance.

Prediction-network input:

    [Siamese distance]

Input dimension:

    1

Trainable parameters:

    16,531

### M2: Siamese+Lab

M2 combines the learned Siamese distance with the six normalized CIELAB coordinates.

Prediction-network input:

    [
        Siamese distance,
        weighted L1*,
        weighted a1*,
        weighted b1*,
        weighted L2*,
        weighted a2*,
        weighted b2*
    ]

The direct Lab values are multiplied by:

    0.01

Input dimension:

    7

Trainable parameters:

    16,723

### M3: Siamese+ΔE

M3 combines the learned Siamese distance with normalized ΔE.

Prediction-network input:

    [
        Siamese distance,
        normalized ΔE
    ]

Input dimension:

    2

Trainable parameters:

    16,563

### M4: Siamese+ΔE+Lab

M4 combines all available features.

Prediction-network input:

    [
        Siamese distance,
        normalized ΔE,
        weighted L1*,
        weighted a1*,
        weighted b1*,
        weighted L2*,
        weighted a2*,
        weighted b2*
    ]

The direct Lab values are multiplied by:

    0.01

Input dimension:

    8

Trainable parameters:

    16,755

## Preprocessing

The six CIELAB input dimensions are normalized independently using feature-wise min-max normalization.

For a feature `x`:

    x_normalized = (x - x_min) / (x_max - x_min)

For Experiments 1 and 2, the normalization limits are calculated from D1.

For Experiment 3, all normalization limits are calculated from D1 and then applied unchanged to D2.

D2 is never used to calculate normalization parameters.

The additional ΔE feature used by M3 and M4 is calculated from the normalized color coordinates and normalized using D1-based limits.

## Training Configuration

The final common training configuration is:

| Parameter | Value |
|---|---:|
| Random seed | 42 |
| Epochs | 300 |
| Batch size | 8 |
| Initial learning rate | 5e-4 |
| Optimizer | Adam |
| Input-noise standard deviation | 0.01 |
| Weight decay | 0 |
| M2 direct-Lab weight | 0.01 |
| M4 direct-Lab weight | 0.01 |
| Loss function | Soft-label categorical cross-entropy |

Gaussian noise is added to the training inputs:

    x_noisy = x + Normal(0, 0.01)

No noise is added during test evaluation.

### Learning-Rate Scheduler

Use `ReduceLROnPlateau`.

The scheduler monitors training loss only.

| Scheduler parameter | Value |
|---|---:|
| Factor | 0.5 |
| Patience | 50 |
| Threshold | 0.001 |
| Minimum learning rate | 1e-6 |

The scheduler is disabled in Experiment 3.

D2 is not used for:

- neural-network training
- learning-rate scheduling
- early stopping
- checkpoint selection
- parameter normalization

## Loss Function

The models are trained using soft-label categorical cross-entropy.

For target distribution `y` and predicted distribution `p`:

    Loss = -mean(sum(y × log(p)))

This loss compares the complete observer-response distribution rather than only the most frequent response category.

Lower loss values indicate better agreement with the observed perceptual-response distributions.

## Experiments

### Experiment 1: D1 Random Split

D1 is randomly divided into:

- 600 training samples
- 200 test samples

The split uses:

    random seed = 42
    test proportion = 0.25

This experiment evaluates model performance when training and test samples come from the same overall dataset distribution.

Notebook:

    notebooks/01_D1_Random_75_25.ipynb

### Experiment 2: D1 Cross-Illumination Evaluation

The model is trained using three D1 illumination conditions and tested using the remaining illumination condition.

The experiment is repeated four times.

This experiment evaluates generalization to an unseen illumination condition.

Notebook:

    notebooks/02_D1_Leave_One_Illumination_Out.ipynb

### Experiment 3: D1-to-D2 Cross-Dataset Evaluation

The model is trained using all 800 D1 samples and tested using all 102 D2 samples.

This experiment evaluates generalization to:

- unseen color pairs
- a different observer group
- a separately collected dataset

Notebook:

    notebooks/03_Full_D1_to_D2.ipynb


## Repository Structure

    low-vision-perceptual-color-difference/
    │
    ├── README.md
    │
    ├── notebooks/
        ├── 01_D1_Random_75_25.ipynb
        ├── 02_D1_Leave_One_Illumination_Out.ipynb
        └── 03_Full_D1_to_D2.ipynb
    

## Requirements

The main dependencies are:

- Python
- PyTorch
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- tabulate
- Jupyter

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. Obtain local copies of D1 and D2.
2. Update the dataset paths in the parameter cell of each notebook.
3. Run the notebooks in numerical order.

The notebooks were designed for execution in Google Colab but can also be adapted to a local Jupyter environment.

## Implementation

The experiments are implemented using PyTorch.

The implementation framework is secondary to the research objective, which is to evaluate perceptual color-difference prediction for people with low vision under different generalization conditions.
