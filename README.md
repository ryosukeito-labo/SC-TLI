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
P(\mathbf{\beta}) =
\lambda_s \left( \alpha \left\| \mathbf{\beta} \right\|_1 + \frac{1-\alpha}{2} \left\| \mathbf{\beta} \right\|_2^2 \right)
+
\lambda_t \sum_{\mathcal{G}_j} \left\| (\mathbf{D} \mathbf{\beta})_{\mathcal{G}_j} \right\|_2
$$

* **Spatial regularization**: L1-L2 penalty (Utsugi, 2019; Ito and Utsugi, 2025)
* **Temporal regularization**: Group Lasso penalty on time differences
* **Optimization**: ADMM (Alternating Direction Method of Multipliers; Boyd et al., 2011)

This formulation enables:

* Sparse and stable spatial inversion
* Structural consistency across time steps
* Robust estimation under noisy conditions

---

## 📂 Repository Structure

```
SC-TLI/
│
├── Src/                 # Core Python implementations (forward & inversion)
│   ├── Forward/
│   ├── Inversion/
├── Examples/            # Jupyter notebooks for synthetic experiments (4 cases)
│   ├── Case_1_1/
│   ├── Case_1_2/
│   ├── Case_2_1/
│   └── Case_2_2/
├── Models/              # Synthetic model datasets for synthetic experiments (4 cases)
│   ├── Case_1_1/
│   ├── Case_1_2/
│   ├── Case_2_1/
│   └── Case_2_2/
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

* The `Examples/` directory contains **4 synthetic model experiments** used in the manuscript.
* The `Models/` directory contains all estimated models.

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
