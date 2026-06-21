# KAN Robustness and Scalability Experiments

Experiment code accompanying the paper **"An Investigation into the Scalability and Robustness of Kolmogorov-Arnold Networks"**, covering Chapters 4–6.

Three notebooks reproduce the paper's experiments on how KANs scale with input dimensionality, how the number of training samples they require to reach a target $R^2$ scales (curse of dimensionality), and how sensitive they are to Gaussian and outlier noise.

---

## Experiments

### Chapter 4 — Effect of Dimensionality (`effect-of-dimensionality.ipynb`)

Trains Static KAN, Dynamic KAN, and a parameter-matched MLP on three synthetic datasets as input dimension increases from 5 to 1000. Reports final test RMSE and R² across three random seeds.

### Chapter 5 — Curse of Dimensionality (`curse-of-dimensionality.ipynb`)

Uses binary search to find the minimum training set size *N* required to reach R² ≥ 0.95, as dimension varies from 5 to 100 (steps of 5). Compares Static and Dynamic KAN on dataset A and C across 3 differnt seeds.

### Chapter 6 — Effect of Noise (`effect-of-noise.ipynb`)

Two noise regimes:
- **Gaussian noise** — additive noise on both inputs and labels, scaled as a fraction of the dataset's standard deviation (σ ∈ {0, 1%, 2%, 5%, 10%, 20%, 50%, 100%} of std).
- **Extreme/outlier noise** — a percentage of labels are replaced with the function's maximum value (0%, 0.5%, 1%, 2%, 5% of the training set). 
Experiments run on datasets A (dim=5), C (dim=5), and C (dim=100) across 3 different seeds.

---

## Datasets

Three synthetic regression functions defined in `utilities.py`:

| Dataset | Function | Notes |
|---------|----------|-------|
| **A** | $\exp\!\left(\frac{1}{\sqrt{d}}\sum_i \left(\sin^2\!\frac{\pi x_i}{2} - \frac{1}{2}\right)\right)$ | Non-linear, compositional |
| **B** | $\frac{1}{\sqrt{d}}\sum_i (\phi(x_i)-0.222)$ where $\phi$ is a bump function | Smooth, non-analytic |
| **C** | $\frac{1}{\sqrt{d}}\sum_i x_i$ | Simple, linear baseline |

All inputs are sampled uniformly from $[-1, 1]^d$.

---

## Models

| Model | Description |
|-------|-------------|
| **Static KAN** | Fixed spline grid (size 3), trained for all 200 epochs |
| **Dynamic KAN** | Grid refined progressively through sizes [3, 5, 10, 20, 50, 100], epochs split evenly |
| **MLP** | Single hidden layer, width chosen to approximately match KAN parameter count |

All models use L-BFGS optimisation with a strong Wolfe line search.

---

## Project Structure

```
.
├── effect-of-dimensionality.ipynb   # Experiments from Chapter 4
├── curse-of-dimensionality.ipynb    # Experiments from Chapter 5
├── effect-of-noise.ipynb            # Experiments from Chapter 6
├── utilities.py                     # Dataset generation and training helpers
```

---

## Setup

Requires Python ≥ 3.13. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

This installs all dependencies including [pykan](https://github.com/KindXiaoming/pykan) from source.

Alternatively, install with pip:

```bash
pip install -r requirements.txt
```

---

## Running the Experiments

Open and run each notebook in order. Results are saved automatically to the corresponding results directory; re-running a notebook will overwrite existing results if hyperparameters are changed.

Training is compute-intensive at higher dimensions. The dimensionality and curse-of-dimensionality notebooks support `xpu`/`cuda` device selection at the top of each notebook.

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Training samples ($N$) | 1000 (Ch.4, Ch.6) / 2000 max (Ch.5) |
| Test samples | 1000 |
| Training epochs | 200 |
| Spline order ($k$) | 3 |
| Learning rate | 1.0 |
| Seeds | 0, 1, 42 |
| Dimensions (Ch.4) | 5, 10, 20, 50, 100, 200, 500, 1000 |
| Dimensions (Ch.5) | 5, 10, 15, …, 100 |