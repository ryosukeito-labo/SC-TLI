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
\lambda_s \left( \alpha \left\| \boldsymbol{\beta} \right\|_1 + \frac{1-\alpha}{2} \left\| \mathbf{\beta} \right\|_2^2 \right)
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
├── src/                 # Core Python implementations (forward & inversion)
│   ├── forward/
│   ├── inversion/
├── examples/            # Jupyter notebooks for synthetic experiments (4 cases)
│   ├── case_1_1/
│   ├── case_1_2/
│   ├── case_2_1/
│   └── case_2_2/
├── README.md
└── requirements.md

```

---
## ▶️ How to Run

1. Clone this repository:

```bash
git clone https://github.com/your-username/SC-TLI.git
cd SC-TLI
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook from the repository root:

```bash
jupyter notebook
```

4. Open and run the notebooks in the `examples/` directory.

---

## 🧪 Experiments

* The `examples/` directory contains **4 synthetic model experiments** used in the manuscript.

---

## ⚙️ Requirements

* Python 3.10
* NumPy
* Jupyter Notebook

(Optional for performance)

* multiprocessing / concurrent.futures

---

## ⚠️ Notes

* Please run notebooks from the repository root directory.
* This code has primarily been tested in a Linux-based Python environment.
* Compatibility with macOS has not been fully verified.
* Large-scale problems may require significant memory and computational resources.

---

## ✍️ Author

* **Ryosuke Ito**
  Institute for Geothermal Sciences,
  Graduate School of Science,
  Kyoto University
  
  Email: ito.ryosuke.22n [at] st.kyoto-u.ac.jp

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
