# 🎬 Movies Rating Prediction

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/notebook-Jupyter-orange)](https://jupyter.org/)
[![Status](https://img.shields.io/badge/status-ready-success)](#)


## 📘 Overview

This repository contains a **Movie Rating Prediction** project built using Python and Jupyter Notebook. The goal is to predict movie ratings using machine learning techniques and demonstrate a complete data science workflow: data collection, cleaning, feature engineering, EDA, model training, evaluation, and inference.


## ⭐ Key Features

* Data cleaning and preprocessing pipeline
* Exploratory Data Analysis (EDA) with visualizations
* Feature engineering for text & numerical fields
* Machine Learning models (Linear Regression, Random Forest, Gradient Boosting)
* Evaluation metrics (RMSE, MAE, R²)
* Reproducible notebook with detailed explanations


## 📁 Files

* `Movies_Rating_Prediction.ipynb` — Main workflow notebook
* `csv's` — Raw/processed datasets folder
* `README.md` — Documentation file
* `LICENSE` — Project license


## 🚀 Quickstart — Run Locally

1. **Clone the repo**:

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
venv\Scripts\activate        # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Launch the notebook**:

```bash
jupyter notebook Movies_Rating_Prediction.ipynb
```

## 🌐 Quickstart — Google Colab

You can upload the notebook to Google Colab :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📊 Dataset : https://grouplens.org/datasets/movielens/100k/

What i have used here are u.data, u.item and u.info 

Typical features:

* `movieId`, `title`, `genres`
* `userId`, `rating`, `timestamp`
* Metadata: `director`, `cast`, `runtime`, `release_year`, `budget`


## 🤖 Model & Evaluation

The notebook includes several ML models evaluated using:

* **RMSE** (Root Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **R²** (Coefficient of Determination)

Sample table (replace with real results):

| Model             | RMSE | MAE  | R²   |
| ----------------- | ---- | ---- | ---- |
| Baseline (Mean)   | 1.45 | 1.10 | 0.00 |
| Linear Regression | 1.15 | 0.90 | 0.12 |
| Random Forest     | 0.98 | 0.75 | 0.35 |


## 🔁 Reproducibility

To reproduce training exactly:
* Use identical train/test split

Example seed code:

```python
import random
import numpy as np
import os

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
```

## 🔮 Tips & Next Steps

* Try embeddings for text fields (TF-IDF, Word2Vec, transformers)
* Add model stacking / ensembling
* Genre-specific prediction models
* Deploy prediction API using Flask or FastAPI


## 🤝 Contributing

Contributions welcome!
Create issues or pull requests. 

## 📄 License

Licensed under the MIT License. See `LICENSE` for details.

## 📬 Contact

Created by **Satyam Bhagat**
