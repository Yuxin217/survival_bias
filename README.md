# Assessing the robustness of heterogeneous treatment effects in survival analysis under informative

## Abstract

Dropout is common in clinical studies, with up to half of patients leaving early due to side effects or other reasons. When dropout is informative (i.e., dependent on survival time), it introduces censoring bias, because of which treatment effect estimates are also biased. In this paper, we propose an assumption-lean framework to assess the robustness of conditional average treatment effect (CATE) estimates in survival analysis when facing censoring bias. Unlike existing works that rely on strong assumptions, such as non-informative censoring, to obtain point estimation, we use partial identification to derive informative bounds on the CATE. Thereby, our framework helps to identify patient subgroups where treatment is effective despite informative censoring. We further propose a novel model-agnostic meta-learner, called \textbf{SurvB-learner}, to estimate the bounds that can be used in combination with arbitrary machine-learning models, and that has favorable theoretical properties such as double-robustness and quasi-oracle efficiency. We finally demonstrate the effectiveness of our meta-learner across various experiments using both simulated and real-world data.

---

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

---

### Datasets

Before running the experiments, the real-world adjuvant dataset needs to be downloaded from the original paper: https://github.com/cancer-oncogenomics/minerva-adjuvant-nsclc

Once downloaded, place the dataset files in the `data/adjuvant_data/` folder.

## Folder Structure

```
auditing_censoring_bias/
├── README.md
├── requirements.txt
├── data/
│   ├── adjuvant_data/                  # Put downloaded dataset here
│   └── synthetic_data/                 # Generated synthetic datasets
├── figures/                            # Generated plots
├── results/                            # Experimental results
├── runnables/                          # Jupyter notebooks
│   ├── adjuvant_data.ipynb
│   └── synthetic_data.ipynb
└── src/                                # Source code
    ├── adjuvant_utils.py
    ├── exp_utils.py
    ├── load_utils.py
    ├── plot_utils.py
    ├── run_utils.py
    └── synthetic_data_generation.py
```

## Reproducing Experiments

All commands should be run from the `code/` directory.

### Synthetic Data Experiments (Table 1 and Figure 3)

Runs Steps 1–3: data generation → SurvB-learner → RMSE table + figure.

```bash
python runnables/synthetic_data.py
```

To run individual steps:

```python
from runnables.synthetic_data import generate_data, run_rct, report_rct, plot_rct

generate_data()   # Step 1
run_rct()         # Step 2
report_rct()      # Table 1
plot_rct()        # Figure 3
```

### Real-Data Analysis (ADJUVANT Trial)

Open and run `runnables/adjuvant_data.ipynb` in Jupyter.

Place the ADJUVANT dataset at `data/adjuvant_data/` before running.

