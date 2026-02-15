# Affine Gaussian Flow (Diagonal Implementation)

## 1. Overview

This project implements a **diagonal scaling transformation**:

\[
x = s \odot z + b
\]

where:

- \( z \sim \mathcal{N}(0, I) \)
- \( s \in \mathbb{R}^d \) is a learnable scale vector
- \( b \in \mathbb{R}^d \) is a learnable bias
- \( \odot \) denotes element-wise multiplication

This corresponds to a **Gaussian distribution with diagonal covariance**.

---

## 2. Theoretical Background

### 2.1 Change of Variables Formula

For an invertible transformation \( x = f(z) \):

\[
\log p_X(x)
=
\log p_Z(z)
-
\log \left| \det J_f(z) \right|
\]

---

### 2.2 Jacobian of the Diagonal Affine Transformation

Since:

\[
f(z) = s \odot z + b
\]

The Jacobian is diagonal:

\[
J_f = \text{diag}(s)
\]

Therefore:

\[
\det J_f = \prod_{i=1}^d s_i
\]

and:

\[
\log |\det J_f|
=
\sum_{i=1}^d \log |s_i|
\]

This makes computation efficient and numerically stable.

---

### 2.3 Inverse Transformation

Because the transformation is element-wise:

\[
z = (x - b) \oslash s
\]

where \( \oslash \) denotes element-wise division.

---

### 2.4 Log-Density Expression

Base density:

\[
\log p_Z(z)
=
- \frac{1}{2}
\left(
\|z\|^2
+
d \log(2\pi)
\right)
\]

Final log-likelihood:

\[
\log p_X(x)
=
\log p_Z(z)
-
\sum_{i=1}^d \log |s_i|
\]

This corresponds to:

\[
x \sim \mathcal{N}(b, \text{diag}(s^2))
\]

---

## 3. Implementation Details

### 3.1 Parameterization

In the current implementation:

```python
self.log_scale = nn.Parameter(torch.zeros(dim))
self.bias = nn.Parameter(torch.zeros(dim))
```

We parameterize scale in log-space:

\[
s = \exp(\text{log\_scale})
\]

This guarantees:

- Positive scale
- Stable determinant computation
- Stable training

---

### 3.2 Forward (Sampling)

```python
def forward(self, z):
    scale = torch.exp(self.log_scale)
    return z * scale + self.bias
```

---

### 3.3 Inverse (Density Evaluation)

```python
def inverse(self, x):
    scale = torch.exp(self.log_scale)
    return (x - self.bias) / scale
```

---

### 3.4 Log-Determinant

```python
log_det = self.log_scale.sum()
```

Since:

\[
\log |\det J_f| = \sum_i \log s_i
\]

and:

\[
\log s_i = \text{log\_scale}_i
\]

---

### 3.5 Log Probability

```python
def log_prob(self, x):
    z = self.inverse(x)
    log_base = -0.5 * (
        z.pow(2).sum(dim=1)
        + dim * math.log(2 * math.pi)
    )
    log_det = self.log_scale.sum()
    return log_base - log_det
```

---

## 4. To go further

### Advantages

- No matrix inversion
- No \( O(d^3) \) determinant computation
- Numerically stable
- Scales to higher dimensions
- Matches diagonal Gaussian dataset assumption

### Limitation

Cannot model correlated dimensions.
