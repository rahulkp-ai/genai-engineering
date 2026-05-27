# Foundations

> ML fundamentals and data engineering — the practical groundwork before deep learning and GenAI.

This section covers applied machine learning and data analysis work corresponding to the early stages of the IBM Generative AI Engineering curriculum. The emphasis here is on **clean, reusable pipelines** rather than tutorial exercises.

---

## Contents

### `ml-experiments/`

End-to-end ML pipelines built with scikit-learn.

| File                            | What It Does                                                                                                                  |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `regression_pipeline.py`        | Modular regression pipeline with preprocessing, feature engineering, cross-validation, and evaluation metrics                 |
| `classification_experiments.py` | Multi-model classification comparison (Logistic Regression, SVM, Random Forest, Gradient Boosting) with consistent evaluation |
| `model_evaluation.ipynb`        | Visual analysis of model performance — confusion matrices, ROC curves, learning curves                                        |

### `data-analysis/`

Structured EDA and data preprocessing utilities.

| File                     | What It Does                                                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| `eda_workflow.ipynb`     | Systematic exploratory analysis — distributions, correlations, outliers, missing data patterns |
| `preprocessing_utils.py` | Reusable preprocessing functions: scalers, encoders, imputers, pipeline builders               |

---

## Engineering Decisions

**Why `.py` files alongside notebooks?**
Notebooks are good for exploration. Production code lives in `.py` files. The `preprocessing_utils.py` module is designed to be imported — it is not a script. The same functions used in the notebook can be imported directly into the deep learning and RAG pipelines upstream.

**Why scikit-learn pipelines over standalone transformers?**
`sklearn.pipeline.Pipeline` enforces fit/transform separation, prevents data leakage during cross-validation, and produces serialisable objects. This is how preprocessing is done in real ML systems.

---

## Setup

```bash
# From repo root
pip install -r requirements.txt

# Run a specific experiment
python foundations/ml-experiments/regression_pipeline.py

# Launch a notebook
jupyter notebook foundations/data-analysis/eda_workflow.ipynb
```

---

## Relationship to the Rest of This Repository

The preprocessing patterns established here — normalization, encoding, train/val/test splits — carry forward directly into:

- [`deep-learning/`](../deep-learning/) — same data preparation logic applied to neural network inputs
- [`rag-and-agents/rag-system/`](../rag-and-agents/rag-system/) — document preprocessing follows the same pipeline structure

The mathematical foundations for the models used here are explored from scratch in the companion repository: [`ann-foundation`](https://github.com/rahulkp-ai/ann-foundation).

---

_Part of the [`genai-engineering`](https://github.com/rahulkp-ai/genai-engineering) portfolio._
