import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

DATA_PATH = Path('notebooks/data/clustered_data.csv')
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

EDUCATION_MAP = {
    'Basic': 0,
    '2n Cycle': 1,
    'Graduation': 2,
    'Master': 3,
    'PhD': 4,
}


def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    # Create a binary target 'will_buy' based on Total_Spending median
    df['will_buy'] = (df['Total_Spending'] > df['Total_Spending'].median()).astype(int)
    df['Education'] = df['Education'].replace(EDUCATION_MAP)
    df = df.fillna(df.mean(numeric_only=True))
    X = df.drop(columns=['cluster', 'will_buy'])
    y = df['will_buy']
    return X, y


def train():
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'logistic_regression': LogisticRegression(max_iter=300, solver='newton-cg'),
        'gradient_boosting': GradientBoostingClassifier(),
    }

    results = []
    for name, model in models.items():
        print('Training', name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
        results.append({'model': name, 'accuracy': acc, 'roc_auc': auc})
        joblib.dump({'model': model, 'features': X.columns.tolist()}, MODELS_DIR / f'{name}.pkl')
        print(f"Saved {name}.pkl")

    # choose best by roc_auc (fallback to accuracy)
    df_res = pd.DataFrame(results)
    if df_res['roc_auc'].notnull().any():
        best = df_res.sort_values('roc_auc', ascending=False).iloc[0]['model']
    else:
        best = df_res.sort_values('accuracy', ascending=False).iloc[0]['model']

    # save best explicitly as best_model.pkl
    best_artifact = joblib.load(MODELS_DIR / f"{best}.pkl")
    joblib.dump(best_artifact['model'], MODELS_DIR / 'best_model.pkl')
    joblib.dump(best_artifact['features'], MODELS_DIR / 'best_features.pkl')

    df_res.to_csv(MODELS_DIR / 'metrics.csv', index=False)
    print('Training complete. Best model:', best)


if __name__ == '__main__':
    train()
