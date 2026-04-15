# SC-TLI (Structurally Coupled Time-Lapse Inversion)

**SC-TLI (Structurally Coupled Time-Lapse Inversion)** is a novel inversion framework for time-lapse magnetic data that integrates spatial L1–L2 regularization with temporal structural coupling.

This repository contains the implementation used in the accompanying manuscript (currently under submission), including synthetic experiments and model datasets.

---

## 📌 Status

⚠️ **Under submission**

This repository is currently synchronized with a manuscript that is under review.
The code and datasets are provided for transparency and reproducibility, but **please do not use or cite this work until the paper is officially published**.

The repository will be updated and fully released upon acceptance.

---

## 📖 Method Overview

The SC-TLI method solves the following optimization problem:

$$
P(\boldsymbol{\beta}) =
\lambda_s \left( \alpha |\boldsymbol{\beta}|_1 + \frac{1-\alpha}{2} \left\| \boldsymbol{\beta} \right\|_2^2 \right)
+
\lambda_t \sum_{\mathcal{G}_j} | (D\boldsymbol{\beta})_{\mathcal{G}_j} |_2
$$

* **Spatial regularization**: Elastic Net (L1 + L2)
* **Temporal regularization**: Group Lasso on time differences
* **Optimization**: ADMM (Alternating Direction Method of Multipliers)

This formulation enables:

* Sparse and stable spatial inversion
* Structural consistency across time steps
* Robust estimation under noisy conditions

---

## 📂 Repository Structure

```
SC-TLI/
│
├── *.ipynb              # Jupyter notebooks for synthetic experiments (12 cases)
├── Models/              # Synthetic model datasets used in the paper
├── *.py                 # Core Python implementations (forward & inversion)
└── README.md
```

---

## ▶️ How to Run

1. Clone or download this repository:

   ```bash
   git clone https://github.com/your-username/SC-TLI.git
   ```

2. Open the Jupyter Notebook files (`.ipynb`)

3. Run the notebooks in a Python environment

---

## 🧪 Experiments

* The repository includes **12 synthetic model experiments** used in the manuscript.
* Each notebook reproduces results corresponding to figures and tables in the paper.
* The `Models/` directory contains all required input models.

---

## ⚙️ Requirements

* Python 3.10
* NumPy
* Jupyter Notebook

(Optional for performance)

* multiprocessing / concurrent.futures

---

## ⚠️ Notes

* This code has primarily been tested in a Linux-based Python environment.
* Compatibility with macOS has not been fully verified.
* Large-scale problems may require significant memory and computational resources.

---

## ✍️ Author

* **Ryosuke Ito**
  Institute for Geothermal Sciences
  Graduate School of Science
  Kyoto University

---

## 📄 License

This project is licensed under the MIT License.
See the LICENSE file for details.

---

## 📬 Contact

For questions regarding the method or implementation, please contact the author.

---

## 🔜 Future Updates

* Public release after paper acceptance
* Documentation improvements
* Additional benchmarks and real-data applications

---
