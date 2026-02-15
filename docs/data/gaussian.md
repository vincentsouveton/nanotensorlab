# Gaussian Dataset Generator

## 1. Overview

The Gaussian dataset generator provides synthetic data sampled from a multivariate normal distribution.
It is used as a controlled benchmark to validate density estimators and generative models.

This dataset is intentionally simple, analytically tractable, and fully parameterizable.

---

## 2. Theoretical Background

### 2.1 Multivariate Gaussian Distribution

We consider a multivariate normal distribution:

\[
x \sim \mathcal{N}(\mu, \Sigma)
\]

where:

- \( \mu \in \mathbb{R}^d \) is the mean vector
- \( \Sigma \in \mathbb{R}^{d \times d} \) is a symmetric positive definite covariance matrix
- \( d \) is the dimensionality

In the current implementation, we restrict to an **isotropic Gaussian**:

\[
\Sigma = \sigma^2 I
\]

which gives:

\[
x = \mu + \sigma \varepsilon
\quad \text{with} \quad
\varepsilon \sim \mathcal{N}(0, I)
\]

---

### 2.2 Statistical Properties

For this distribution:

\[
\mathbb{E}[x] = \mu
\]

\[
\text{Cov}(x) = \sigma^2 I
\]

These closed-form expressions allow:

- Analytical validation of learned parameters
- Verification of convergence
- Sanity checks for generative models

---

### 2.3 Role in the Project

The Gaussian dataset serves as:

- A controlled test environment
- A debugging benchmark
- A validation reference for maximum likelihood training
- A baseline case for flows and parametric estimators

Because the true distribution is known, performance can be measured precisely.

---

## 3. Technical Implementation

### 3.1 Design Principles

The dataset generator is:

- Independent from models
- Stateless (except for parameters)
- Deterministic under fixed random seed
- Fully configurable via YAML

Typical structure:

```
data/generators/gaussian.py
```

---

### 3.2 Parameterization

Example YAML configuration:

```yaml
name: gaussian

params:
  dim: 2
  mean: 0.0
  std: 1.0
  n_samples: 10000
```

Parameters:

- `dim`: dimensionality \( d \)
- `mean`: scalar or vector mean
- `std`: standard deviation \( \sigma \)
- `n_samples`: dataset size

---

### 3.3 Sampling Mechanism

Sampling is implemented as:

\[
x = \mu + \sigma \cdot \text{torch.randn}(n, d)
\]

This ensures:

- Exact Gaussian samples
- Vectorized computation
- Efficient GPU compatibility
- Reproducibility via global seed

---

### 3.4 Integration in the Training Pipeline

The complete pipeline is:

```
YAML â Factory â Generator â TensorDataset â DataLoader
```

The dataset layer is completely independent from the model layer.
This separation guarantees modularity, reproducibility, and clean experimentation.

---

## 4. Limitations and Extensions

### Current limitations

- Diagonal covariance only
- Single unimodal Gaussian
- No structured correlations

### Possible extensions

- Full covariance matrix support
- Gaussian mixtures
- Heavy-tailed distributions
- Non-Gaussian synthetic benchmarks

---

## 5. Summary

The Gaussian dataset generator provides a mathematically transparent and computationally simple benchmark.

Because the true distribution is known analytically, it enables precise validation of:

- Maximum likelihood estimators
- Normalizing flows
- Sampling correctness
- Log-likelihood computations

It serves as the foundational testing ground for the entire generative modeling pipeline.
