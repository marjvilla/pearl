"""
risk_predictor.py
-----------------
Maps patient data from the SQLite database to the feature space expected by
preeclampsia_final_xgb.pkl, then returns a risk prediction.

Required files (produced by Train_model_updated.py):
    preeclampsia_final_xgb.pkl
    final_feature_columns.pkl
    raw_features.pkl
    best_threshold.pkl
"""

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH           = "preeclampsia_final_xgb.pkl"
FEATURE_COLUMNS_PATH = "final_feature_columns.pkl"
RAW_FEATURES_PATH    = "raw_features.pkl"
THRESHOLD_PATH       = "best_threshold.pkl"
FEATURE_DISPLAY_NAMES_PATH = "feature_display_names.pkl"

# ── Race mapping ───────────────────────────────────────────────────────────────
# Maps the string stored in the UI's raceComboBox to the numeric codes used in
# training.  Adjust the keys to match whatever your combobox contains.
RACE_MAP = {
    "White":                    1,
    "Black / African American": 2,
    "Asian":                    3,
    "Other":                    4,
    "Unknown":                  np.nan,
}

# ── Risk-factor strings (must match complete_registration in main.py) ──────────
RF_PREGEST_DM  = "Pregestational Diabetes"
RF_OTHER_DM    = "Gestational / Other Diabetes"
RF_CHRON_HTN   = "Chronic Hypertension"
RF_SMOKE_EVER  = "Ever Used Tobacco"
RF_SMOKE_3MO   = "Smoked Tobacco in the 3 Months Prior to Pregnancy"




# ── Patient-specific explanation helpers ──────────────────────────────────────
def _readable_feature_name(feature: str) -> str:
    """Convert model column names into UI-friendly labels."""
    name_map = {
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

    if feature.startswith("CRace_"):
        return "Race category"

    return name_map.get(feature, feature)


def _get_top_3_feature_contributions(
    model,
    X_aligned: pd.DataFrame,
    feature_columns: list,
    X_raw: pd.DataFrame,
    visits_used: int,
) -> list:
    """
    Return the top 3 patient-specific feature contributions for this prediction.
    Hides unavailable future-visit features and unchecked/missing medical
    conditions so the UI explanation only displays factors the patient actually has.
    """
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_aligned, feature_names=feature_columns)

    # Last column is the model bias/base value, not a patient feature.
    contributions = booster.predict(dmatrix, pred_contribs=True)[0][:-1]

    contribution_df = pd.DataFrame({
        "feature": feature_columns,
        "contribution": contributions,
        "absolute_contribution": np.abs(contributions),
    })

    medical_condition_features = {
        "HTN_V1",
        "Diabetes_V1",
        "GestationalDiabetes_V2",
        "Hyperthyroid_V1",
        "Hypothyroid_V1",
        "KidneyDisease_V1",
        "APA_Thrombophilia_V1",
        "SLE_V1",
    }

    def patient_has_feature(feature_name: str) -> bool:
        if feature_name not in X_raw.columns:
            return True
        val = X_raw.iloc[0][feature_name]
        return pd.notna(val) and float(val) == 1.0

    def hide_from_ui(feature_name: str) -> bool:
        # Do not show missingness indicators as "top factors" in the UI.
        if feature_name.endswith("_missing"):
            return True

        # Do not show Visit 2 or Visit 3 factors before those visits exist.
        if visits_used < 2 and "V2" in feature_name:
            return True
        if visits_used < 3 and "V3" in feature_name:
            return True

        # Do not show unchecked/missing medical conditions as top factors.
        if feature_name in medical_condition_features and not patient_has_feature(feature_name):
            return True

        # Do not show medical-history interaction features unless the condition is checked.
        if feature_name == "ChronHTN_x_MAP" and not patient_has_feature("HTN_V1"):
            return True
        if feature_name == "Diabetes_x_MAP" and not (
            patient_has_feature("Diabetes_V1") or patient_has_feature("GestationalDiabetes_V2")
        ):
            return True

        return False

    contribution_df = contribution_df[
        ~contribution_df["feature"].astype(str).apply(hide_from_ui)
    ]

    contribution_df = contribution_df.sort_values(
        "absolute_contribution",
        ascending=False,
    ).head(3)

    top_features = []
    for _, row in contribution_df.iterrows():
        contribution = float(row["contribution"])
        direction = "increased risk" if contribution > 0 else "decreased risk"
        top_features.append({
            "feature": _readable_feature_name(str(row["feature"])),
            "direction": direction,
            "contribution": contribution,
        })

    return top_features

def _flag(risk_factors: list, label: str) -> float:
    """Return 1.0 if *label* is in the risk-factor list, else NaN (missing)."""
    if label in risk_factors:
        return 1.0
    # We genuinely don't know → let the model's missingness indicator handle it
    return np.nan


def _bmi(weight_lbs: float, height_inches: int) -> float:
    """Imperial BMI formula.  Returns NaN when inputs are invalid."""
    if not weight_lbs or not height_inches or height_inches <= 0:
        return np.nan
    return round((weight_lbs / (height_inches ** 2)) * 703, 1)


def build_feature_row(patient: dict, risk_factors: list, bp_readings: list) -> pd.DataFrame:
    """
    Construct a single-row DataFrame whose columns match the raw features that
    were fed into pd.get_dummies() during training.

    Parameters
    ----------
    patient     : sqlite3.Row (or dict) from the patients table
    risk_factors: list of strings from the risk_factors table
    bp_readings : list of sqlite3.Row ordered by timestamp ASC (oldest first)
                  Each row must have: systolic, diastolic, map_value, weight,
                  gest_age_at_visit, new_age
    """

    # ── Unpack up to 3 visits ─────────────────────────────────────────────────
    v = [dict(r) if hasattr(r, 'keys') else r for r in bp_readings[:3]]
    v1 = v[0] if len(v) >= 1 else {}
    v2 = v[1] if len(v) >= 2 else {}
    v3 = v[2] if len(v) >= 3 else {}

    def bp(visit, key):
        val = visit.get(key) if visit else None
        return float(val) if val is not None else np.nan

    # ── Derived longitudinal features ─────────────────────────────────────────
    sys_vals  = [bp(x, 'systolic')   for x in v if x]
    dia_vals  = [bp(x, 'diastolic')  for x in v if x]
    map_vals  = [bp(x, 'map_value')  for x in v if x]

    def safe(vals, fn):
        clean = [x for x in vals if not np.isnan(x)]
        return fn(clean) if clean else np.nan

    mean_map  = safe(map_vals, np.mean)
    max_map   = safe(map_vals, max)
    map_range = (max(map_vals) - min(map_vals)) if len([x for x in map_vals if not np.isnan(x)]) >= 2 else np.nan

    mean_sys  = safe(sys_vals, np.mean)
    max_sys   = safe(sys_vals, max)
    sys_range = (max(sys_vals) - min(sys_vals)) if len([x for x in sys_vals if not np.isnan(x)]) >= 2 else np.nan

    mean_dia  = safe(dia_vals, np.mean)
    max_dia   = safe(dia_vals, max)
    dia_range = (max(dia_vals) - min(dia_vals)) if len([x for x in dia_vals if not np.isnan(x)]) >= 2 else np.nan

    def delta(a, b):
        return (b - a) if not np.isnan(a) and not np.isnan(b) else np.nan

    map_d21  = delta(bp(v1, 'map_value'), bp(v2, 'map_value'))
    map_d32  = delta(bp(v2, 'map_value'), bp(v3, 'map_value'))
    map_d31  = delta(bp(v1, 'map_value'), bp(v3, 'map_value'))
    sys_d21  = delta(bp(v1, 'systolic'),  bp(v2, 'systolic'))
    sys_d32  = delta(bp(v2, 'systolic'),  bp(v3, 'systolic'))
    sys_d31  = delta(bp(v1, 'systolic'),  bp(v3, 'systolic'))
    dia_d21  = delta(bp(v1, 'diastolic'), bp(v2, 'diastolic'))
    dia_d32  = delta(bp(v2, 'diastolic'), bp(v3, 'diastolic'))
    dia_d31  = delta(bp(v1, 'diastolic'), bp(v3, 'diastolic'))

    def count_over(vals, thresh):
        clean = [x for x in vals if not np.isnan(x)]
        return float(sum(1 for x in clean if x > thresh)) if clean else np.nan

    def two_or_more(vals, thresh):
        clean = [x for x in vals if not np.isnan(x)]
        return 1.0 if sum(1 for x in clean if x > thresh) >= 2 else 0.0 if clean else np.nan

    # ── Race & ethnicity ──────────────────────────────────────────────────────
    race_val = RACE_MAP.get(patient.get('race', 'Unknown'), np.nan) if patient else np.nan

    # ── Demographics ─────────────────────────────────────────────────────────
    age_v1   = float(v1['new_age']) if v1.get('new_age') is not None else float(patient.get('age', np.nan) or np.nan)
    ga_wks   = float(v1['gest_age_at_visit']) if v1.get('gest_age_at_visit') is not None else float(patient.get('gestational_weeks', np.nan) or np.nan)
    bmi_val  = _bmi(bp(v1, 'weight'), patient.get('height_inches'))

    # ── Risk factors from registration ───────────────────────────────────────
    pre_dm   = 1.0 if RF_PREGEST_DM in risk_factors else 0.0
    other_dm = 1.0 if RF_OTHER_DM   in risk_factors else 0.0
    chron_ht = 1.0 if RF_CHRON_HTN  in risk_factors else 0.0
    smoke1   = 1.0 if RF_SMOKE_EVER in risk_factors else 0.0
    smoke2   = 1.0 if RF_SMOKE_3MO  in risk_factors else 0.0

    # ── Pulse pressure (V1) ───────────────────────────────────────────────────
    pp = delta(bp(v1, 'diastolic'), bp(v1, 'systolic'))    # systolic - diastolic

    # ── Interaction features ──────────────────────────────────────────────────
    v1_map  = bp(v1, 'map_value')
    cxm     = chron_ht * v1_map if not np.isnan(v1_map) else np.nan
    bxm     = bmi_val  * v1_map if not np.isnan(bmi_val) and not np.isnan(v1_map) else np.nan
    dxm     = (pre_dm or other_dm) * v1_map if not np.isnan(v1_map) else np.nan

    # ── Threshold flags ───────────────────────────────────────────────────────
    def flag_over(val, thresh):
        return 1.0 if (not np.isnan(val) and val > thresh) else 0.0 if not np.isnan(val) else np.nan

    # ── Assemble raw-feature dict (matches RAW_FEATURES in training script) ───
    row = {
        # Core demographics
        "GAwks_screen":                ga_wks,
        "Age_at_V1":                   age_v1,
        "BMI":                         bmi_val,
        "PreGestDM":                   pre_dm,
        "oDM":                         other_dm,
        "ChronHTN":                    chron_ht,
        "SmokeCat1":                   smoke1,
        "SmokeCat2":                   smoke2,
        "CRace":                       race_val,
        "Race":                        race_val,
        "eRace":                       race_val,
        "eHispanic":                   float(patient.get('is_hispanic', 0) or 0),

        # Visit 1
        "V1_Avg_Systolic":             bp(v1, 'systolic'),
        "V1_Avg_Diastolic":            bp(v1, 'diastolic'),
        "V1_MAP":                      v1_map,

        # Visit 2
        "V2_Avg_Systolic":             bp(v2, 'systolic'),
        "V2_Avg_Diastolic":            bp(v2, 'diastolic'),
        "V2_MAP":                      bp(v2, 'map_value'),

        # Visit 3
        "V3_Avg_Systolic":             bp(v3, 'systolic'),
        "V3_Avg_Diastolic":            bp(v3, 'diastolic'),
        "V3_MAP":                      bp(v3, 'map_value'),

        # Longitudinal MAP
        "MAP_Delta_V2_V1":             map_d21,
        "MAP_Delta_V3_V2":             map_d32,
        "Max_Recorded_MAP":            max_map,
        "MAP_Delta_V3_V1":             map_d31,
        "Mean_MAP_All_Visits":         mean_map,
        "MAP_Range":                   map_range,
        "Num_MAP_over_90":             count_over(map_vals, 90),
        "MAP_over_90_two_or_more_visits": two_or_more(map_vals, 90),

        # Longitudinal systolic
        "Systolic_Delta_V2_V1":        sys_d21,
        "Systolic_Delta_V3_V2":        sys_d32,
        "Systolic_Delta_V3_V1":        sys_d31,
        "Mean_Systolic_All_Visits":    mean_sys,
        "Max_Systolic_All_Visits":     max_sys,
        "Systolic_Range":              sys_range,
        "Num_Systolic_over_130":       count_over(sys_vals, 130),
        "Systolic_over_130_two_or_more_visits": two_or_more(sys_vals, 130),

        # Longitudinal diastolic
        "Diastolic_Delta_V2_V1":       dia_d21,
        "Diastolic_Delta_V3_V2":       dia_d32,
        "Diastolic_Delta_V3_V1":       dia_d31,
        "Mean_Diastolic_All_Visits":   mean_dia,
        "Max_Diastolic_All_Visits":    max_dia,
        "Diastolic_Range":             dia_range,
        "Num_Diastolic_over_80":       count_over(dia_vals, 80),
        "Diastolic_over_80_two_or_more_visits": two_or_more(dia_vals, 80),

        # Medical history — not captured in current UI → treated as unknown/missing
        "HTN_V1":                      chron_ht,      # best proxy: Chronic HTN flag
        "Diabetes_V1":                 pre_dm,        # best proxy: PreGestDM flag
        "GestationalDiabetes_V2":      other_dm,      # best proxy: Gest/Other DM flag
        "Hyperthyroid_V1":             np.nan,        # not collected → missing
        "Hypothyroid_V1":              np.nan,
        "KidneyDisease_V1":            np.nan,
        "APA_Thrombophilia_V1":        np.nan,
        "SLE_V1":                      np.nan,

        # V1 threshold / interaction features
        "Age_over_35":                 flag_over(age_v1, 35),
        "BMI_over_30":                 flag_over(bmi_val, 30),
        "MAP_over_90":                 flag_over(v1_map, 90),
        "Systolic_over_130":           flag_over(bp(v1, 'systolic'), 130),
        "Diastolic_over_80":           flag_over(bp(v1, 'diastolic'), 80),
        "Pulse_Pressure":              pp,
        "ChronHTN_x_MAP":              cxm,
        "BMI_x_MAP":                   bxm,
        "Diabetes_x_MAP":              dxm,

        # Longitudinal threshold flags
        "V2_MAP_over_90":              flag_over(bp(v2, 'map_value'), 90),
        "V3_MAP_over_90":              flag_over(bp(v3, 'map_value'), 90),
        "Max_MAP_over_90":             flag_over(max_map, 90),
        "MAP_Increased_V2":            1.0 if (not np.isnan(map_d21) and map_d21 > 0) else 0.0 if not np.isnan(map_d21) else np.nan,
        "MAP_Increased_V3":            1.0 if (not np.isnan(map_d32) and map_d32 > 0) else 0.0 if not np.isnan(map_d32) else np.nan,
        "V2_Systolic_over_130":        flag_over(bp(v2, 'systolic'), 130),
        "V3_Systolic_over_130":        flag_over(bp(v3, 'systolic'), 130),
        "V2_Diastolic_over_80":        flag_over(bp(v2, 'diastolic'), 80),
        "V3_Diastolic_over_80":        flag_over(bp(v3, 'diastolic'), 80),
    }

    # ── Add missingness indicators for every raw feature ─────────────────────
    # (The training script added these automatically; we must replicate them.)
    missing_indicator_keys = [
        "GAwks_screen", "Age_at_V1", "BMI", "PreGestDM", "oDM", "ChronHTN",
        "SmokeCat1", "SmokeCat2", "CRace", "Race", "eRace", "eHispanic",
        "V1_Avg_Systolic", "V1_Avg_Diastolic", "V1_MAP",
        "V2_Avg_Systolic", "V2_Avg_Diastolic", "V2_MAP",
        "V3_Avg_Systolic", "V3_Avg_Diastolic", "V3_MAP",
        "MAP_Delta_V2_V1", "MAP_Delta_V3_V2", "Max_Recorded_MAP",
        "HTN_V1", "Diabetes_V1", "GestationalDiabetes_V2",
        "Hyperthyroid_V1", "Hypothyroid_V1", "KidneyDisease_V1",
        "APA_Thrombophilia_V1", "SLE_V1",
    ]
    for key in missing_indicator_keys:
        row[f"{key}_missing"] = 1.0 if np.isnan(row.get(key, np.nan)) else 0.0

    return pd.DataFrame([row])


def predict_risk(patient: dict, risk_factors: list, bp_readings: list) -> dict:
    """
    Run the trained XGBoost model and return a result dict:
        {
            'label':       'High Risk' | 'Low Risk',
            'probability': float  (0–100 %)
            'visits_used': int,
            'missing_bp':  bool,   # True when < 1 reading available
        }

    Raises FileNotFoundError if the model .pkl files are missing.
    """
    # Load saved artefacts
    model           = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    threshold       = joblib.load(THRESHOLD_PATH)

    # Sort readings oldest → newest so V1 = first visit
    sorted_readings = sorted(bp_readings, key=lambda r: r['timestamp'])

    if not sorted_readings:
        return {
            'label':       'Insufficient Data',
            'probability': None,
            'visits_used': 0,
            'missing_bp':  True,
        }

    # Build raw-feature row
    X_raw = build_feature_row(
        patient      = dict(patient),
        risk_factors = risk_factors,
        bp_readings  = sorted_readings,
    )

    # One-hot encode (same as training: pd.get_dummies, drop_first=False)
    X_encoded = pd.get_dummies(X_raw, drop_first=False)

    # Align to the exact column order the model was trained on
    X_aligned = X_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict
    # XGBoost explanations require the exact feature names and numeric values.
    feature_columns = list(feature_columns)
    X_aligned = X_aligned.astype(float)

    prob   = float(model.predict_proba(X_aligned)[0, 1])
    if prob < 0.175:
        label = "Low Risk"
    elif prob < 0.25:
        label = "Moderate Risk"
    else:
        label = "High Risk"

    # Patient-specific top contributors for the UI.
    top_3_features = _get_top_3_feature_contributions(
        model=model,
        X_aligned=X_aligned,
        feature_columns=feature_columns,
        X_raw=X_raw,
        visits_used=len(sorted_readings),
    )

    return {
        'label':          label,
        'probability':    round(prob * 100, 1),
        'visits_used':    len(sorted_readings),
        'missing_bp':     False,
        'top_3_features': top_3_features,
    }
