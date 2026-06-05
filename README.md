<<<<<<< HEAD
# Customer-categorization
=======
Customer Categorization
======================

A small project for customer segmentation, feature engineering, clustering, and classification models. Contains notebooks for EDA and feature engineering, training scripts, a Streamlit app, and saved model artifacts.

Authors
-------

1. Abhinay Pal
2. Abhishek Kumar

Repository structure
--------------------

- [streamlit_app.py](streamlit_app.py): Streamlit front-end to explore results and demo the model.
- [train_models.py](train_models.py): Script to train classification models.
- [models/](models): Trained model artifacts and `metrics.csv` with evaluation results.
- [notebooks/](notebooks): Jupyter notebooks for EDA, feature engineering, clustering, and modeling.
- [data/clustered_data.csv](data/clustered_data.csv): Processed dataset used by notebooks and training.
- [requirements.txt](requirements.txt): Python dependencies.

Requirements
------------

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\\Scripts\\activate    # Windows
pip install -r requirements.txt
```

Running the Streamlit app
-------------------------

Start the app locally (the project includes a `streamlit_8502.err` log used during development):

```bash
streamlit run streamlit_app.py --server.port 8502
```

Open the URL printed by Streamlit (typically http://localhost:8502).

Training models
---------------

To retrain models from the processed data, run:

```bash
python train_models.py
```

Training artifacts and metrics will be written to the `models/` directory (see `models/metrics.csv`).

Notebooks
---------

- [notebooks/EDA.ipynb](notebooks/EDA.ipynb): Exploratory data analysis.
- [notebooks/Feature_engineering_and_clustering.ipynb](notebooks/Feature_engineering_and_clustering.ipynb): Feature engineering and clustering pipeline.
- [notebooks/Feature_Selection_and_classification.ipynb](notebooks/Feature_Selection_and_classification.ipynb): Feature selection and classifier experiments.

Data
----

Place raw or additional datasets in the `data/` folder. The project currently includes `data/clustered_data.csv` used by notebooks.

Notes
-----

- Model training used CatBoost (training logs are under `notebooks/catboost_info/`).
- Metrics from experiments are available in `models/metrics.csv`.

Next steps
----------

- Update this README with specific model descriptions and example outputs.
- Add a requirements section with pinned versions if reproducibility is needed.

---

If you want, I can run the Streamlit app or expand this README with specific model details and usage examples.
>>>>>>> 88c8378 (Initial commit)
