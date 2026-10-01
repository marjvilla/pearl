import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
import joblib

DATA_PATH = "/project/btec-design3/Updated Model/preeclampsia_training_ready.csv"

# Saved model files
MODEL_PATH = "preeclampsia_final_xgb.pkl"
FEATURE_COLUMNS_PATH = "final_feature_columns.pkl"
RAW_FEATURES_PATH = "raw_features.pkl"
THRESHOLD_PATH = "best_threshold.pkl"
SELECTED_FEATURE_SET_PATH = "selected_feature_set.pkl"
SELECTED_WEIGHT_PATH = "selected_scale_pos_weight.pkl"
FEATURE_DISPLAY_NAMES_PATH = "feature_display_names.pkl"

FEATURE_DISPLAY_NAMES = {
    "V1_MAP": "Visit 1 MAP",
    "V2_MAP": "Visit 2 MAP",
    "V3_MAP": "Visit 3 MAP",
    "V1_Avg_Systolic": "Visit 1 systolic blood pressure",
    "V2_Avg_Systolic": "Visit 2 systolic blood pressure",
    "V3_Avg_Systolic": "Visit 3 systolic blood pressure",
    "V1_Avg_Diastolic": "Visit 1 diastolic blood pressure",
    "V2_Avg_Diastolic": "Visit 2 diastolic blood pressure",
    "V3_Avg_Diastolic": "Visit 3 diastolic blood pressure",
    "HTN_V1": "History of hypertension",
    "Diabetes_V1": "History of diabetes",
    "GestationalDiabetes_V2": "Gestational diabetes",
    "Hyperthyroid_V1": "Hyperthyroidism",
    "Hypothyroid_V1": "Hypothyroidism",
    "KidneyDisease_V1": "Kidney disease",
    "APA_Thrombophilia_V1": "Antiphospholipid syndrome / thrombophilia",
    "SLE_V1": "Systemic lupus erythematosus",
    "CRace": "Race category",
    "Race": "Race",
    "eRace": "Encoded race",
    "eHispanic": "Hispanic ethnicity",
}


# Best 80/20 feature-search model settings
SELECTED_FEATURE_SET = "required_only"
SELECTED_SCALE_POS_WEIGHT = 21.288
SELECTED_THRESHOLD = 0.250
RANDOM_STATE = 42

# Required-only feature set:
# race, medical history, MAP, systolic, and diastolic for visits 1, 2, and 3
RAW_FEATURES = [
    # Race / ethnicity
    "CRace",
    "Race",
    "eRace",
    "eHispanic",

    # Visit 1 BP
    "V1_Avg_Systolic",
    "V1_Avg_Diastolic",
    "V1_MAP",

    # Visit 2 BP
    "V2_Avg_Systolic",
    "V2_Avg_Diastolic",
    "V2_MAP",

    # Visit 3 BP
    "V3_Avg_Systolic",
    "V3_Avg_Diastolic",
    "V3_MAP",

    # Medical history
    "HTN_V1",
    "Diabetes_V1",
    "GestationalDiabetes_V2",
    "Hyperthyroid_V1",
    "Hypothyroid_V1",
    "KidneyDisease_V1",
    "APA_Thrombophilia_V1",
    "SLE_V1",
]


def calculate_metrics(y_true, y_pred, probs=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    if probs is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, probs)
        metrics["pr_auc"] = average_precision_score(y_true, probs)

    return metrics


def print_final_metrics(y_true, y_pred, probs=None):
    metrics = calculate_metrics(y_true, y_pred, probs)

    print("\n--- Final Performance ---")
    print(f"Accuracy:              {metrics['accuracy']:.3f}")
    print(f"Precision:             {metrics['precision']:.3f}")
    print(f"Recall/Sensitivity:    {metrics['recall']:.3f}")
    print(f"Specificity:           {metrics['specificity']:.3f}")
    print(f"F1 Score:              {metrics['f1']:.3f}")
    print(f"False Positive Rate:   {metrics['fpr']:.3f}")
    print(f"False Negative Rate:   {metrics['fnr']:.3f}")

    if "roc_auc" in metrics:
        print(f"ROC-AUC:               {metrics['roc_auc']:.3f}")
        print(f"PR-AUC:                {metrics['pr_auc']:.3f}")

    print("\nConfusion Matrix:")
    print("Rows = actual class, columns = predicted class")
    print("              Pred 0   Pred 1")
    print(f"Actual 0       {metrics['tn']:6d}   {metrics['fp']:6d}")
    print(f"Actual 1       {metrics['fn']:6d}   {metrics['tp']:6d}")

    print("\nCounts:")
    print(f"Correct negatives:     {metrics['tn']}")
    print(f"False positives:       {metrics['fp']}")
    print(f"False negatives:       {metrics['fn']}")
    print(f"Correct positives:     {metrics['tp']}")
    print(f"Total correct:         {metrics['tn'] + metrics['tp']} / {len(y_true)}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def train_final_model():
    df = pd.read_csv(DATA_PATH)

    if "target_label" not in df.columns:
        raise ValueError("target_label column not found in cleaned dataset.")

    available_features = [col for col in RAW_FEATURES if col in df.columns]
    missing_features = [col for col in RAW_FEATURES if col not in df.columns]

    if missing_features:
        print("Warning: these model features were not found and will be skipped:")
        for col in missing_features:
            print(f"  - {col}")

    print("\nUsing selected model settings:")
    print(f"Selected feature set:    {SELECTED_FEATURE_SET}")
    print(f"Chosen scale_pos_weight: {SELECTED_SCALE_POS_WEIGHT:.3f}")
    print(f"Chosen threshold:        {SELECTED_THRESHOLD:.3f}")

    print("\nUsing raw features:")
    for col in available_features:
        print(f"  - {col}")

    X_raw = df[available_features].copy()
    y = df["target_label"].astype(int)

    X = pd.get_dummies(X_raw, drop_first=False)

    print(f"\nModel matrix shape after one-hot encoding: {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.01,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=SELECTED_SCALE_POS_WEIGHT,
        eval_metric="logloss",
        missing=np.nan,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= SELECTED_THRESHOLD).astype(int)

    print_final_metrics(y_test, preds, probs=probs)

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\nTop 25 Feature Importances:")
    print(feature_importance.head(25).to_string(index=False))

    feature_importance.to_csv("feature_importance.csv", index=False)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(list(X.columns), FEATURE_COLUMNS_PATH)
    joblib.dump(available_features, RAW_FEATURES_PATH)
    joblib.dump(SELECTED_THRESHOLD, THRESHOLD_PATH)
    joblib.dump(SELECTED_FEATURE_SET, SELECTED_FEATURE_SET_PATH)
    joblib.dump(SELECTED_SCALE_POS_WEIGHT, SELECTED_WEIGHT_PATH)
    joblib.dump(FEATURE_DISPLAY_NAMES, FEATURE_DISPLAY_NAMES_PATH)

    print("\nSaved files:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {FEATURE_COLUMNS_PATH}")
    print(f"  - {RAW_FEATURES_PATH}")
    print(f"  - {THRESHOLD_PATH}")
    print(f"  - {SELECTED_FEATURE_SET_PATH}")
    print(f"  - {SELECTED_WEIGHT_PATH}")
    print(f"  - {FEATURE_DISPLAY_NAMES_PATH}")
    print("  - feature_importance.csv")


if __name__ == "__main__":
    train_final_model()
