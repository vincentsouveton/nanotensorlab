# 🧪 nanotensorlab

**nanotensorlab** is an open-source educational and research-oriented framework
designed to experiment with modern neural models for:

- **Sampling**
- **Regression**

The project focuses on **simple synthetic datasets (1D, 2D, 3D)** to make
advanced machine learning concepts easy to understand, visualize, and extend.

---

## 🎯 Goals

- Provide **clean and minimal implementations**
- Combine **theory and practice**
- Enable **rapid experimentation**
- Foster **collaborative research and education**

---

## 🧩 Main Components

### Data Generation
Synthetic datasets with analytical ground truth:
- 1D / 2D / 3D toy distributions
- Physics simulations
- Controlled noise settings

### Models
- Generative models
- Regression (meta)models
- Classification models

### Training & Evaluation
- Modular training loops
- Metrics and uncertainty estimation
- Sampling diagnostics

### Visualization
- 1D, 2D, 3D plots

### Documentation
- Mathematical foundations
- Implementation details
- Research-oriented explanations

---

## 🚀 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/vincentsouveton/nanotensorlab.git
cd nanotensorlab
```

---

### 2️⃣ Create a virtual environment (recommended)

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Upgrade pip

```bash
pip install --upgrade pip
```

---

### 4️⃣ Install NanoTensorLab in editable mode

```bash
pip install -e .
```

---

## ⚙️ Running an Experiment

```bash
python scripts/train.py --config configs/XXX.yaml
```

Best and last model checkpoints are automatically saved inside:

```
outputs/
```

---

## 📌 How to Cite

If you use **NanoTensorLab** in academic work, research, or teaching material,
please cite it as follows:

### BibTeX
```bibtex
@software{nanotensorlab,
  title  = {nanotensorlab: A minimalist lab for neural sampling and regression},
  author = {{Vincent Souveton}},
  year   = {2026},
  url    = {https://github.com/vincentsouveton/nanotensorlab}
}
```

You may also cite it in plain text as:

> nanotensorlab, an open-source educational framework for neural modeling.

---

## 📜 License

This project is released under the **MIT License**.
See the `LICENSE` file for details.
