from io import BytesIO
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="LiveWell+", page_icon="👤", layout="wide")

CURRENT_YEAR = 2026
TARGET_STATE = "California"
TARGET_PATIENT_COUNT = 10000
DEFAULT_DEMO_PATIENT_ID = "4f6f3c94-9832-e91e-a5bd-bc482f3b1019"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR_CANDIDATES = [
    APP_DIR / "app_data",
    APP_DIR,
    APP_DIR / "synthea-4.0.0" / "synthea-4.0.0" / "output" / "csv",
]


def apply_theme(theme_name: str) -> None:
    if theme_name == "Light":
        bg = "#f8fafd"
        panel = "#ffffff"
        sidebar = "#e8f0fb"
        text = "#16304f"
        muted = "#62758f"
        border = "#d6e1f0"
        accent = "#1677d2"
        accent_soft = "#eff7ff"
        success_bg = "#e8f7ef"
        success_text = "#166b46"
        warning_bg = "#fff5df"
        warning_text = "#9a6200"
        error_bg = "#fdeaea"
        error_text = "#ab2f2f"
    else:
        bg = "#0b1220"
        panel = "#121b2d"
        sidebar = "#141d31"
        text = "#f8fafc"
        muted = "#b6c2d2"
        border = "#2a3854"
        accent = "#3b82f6"
        accent_soft = "#1c2d52"
        success_bg = "#123524"
        success_text = "#86efac"
        warning_bg = "#3b2f10"
        warning_text = "#fcd34d"
        error_bg = "#3b1715"
        error_text = "#fca5a5"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {bg};
            color: {text};
        }}
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        #MainMenu,
        footer {{
            visibility: hidden;
            height: 0;
            position: fixed;
        }}
        html, body, [class*="css"]  {{
            color: {text};
        }}
        [data-testid="stSidebar"] {{
            background: {sidebar};
        }}
        [data-testid="stSidebar"] * {{
            color: {text} !important;
        }}
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {{
            color: {text} !important;
        }}
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {{
            background: {panel} !important;
            border: 1px solid {border} !important;
            color: {text} !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: {panel} !important;
            border: 1px dashed {border} !important;
        }}
        .stButton > button,
        .stDownloadButton > button,
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-secondaryFormSubmit"] {{
            background: {accent} !important;
            color: white !important;
            border: 1px solid {accent} !important;
            border-radius: 14px !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 20px rgba(39, 100, 255, 0.18) !important;
        }}
        .stButton > button:hover,
        .stDownloadButton > button:hover,
        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="baseButton-secondary"]:hover,
        [data-testid="baseButton-secondaryFormSubmit"]:hover {{
            background: #0b5ca0 !important;
            border-color: #0b5ca0 !important;
            color: white !important;
        }}
        .stButton > button:focus,
        .stDownloadButton > button:focus,
        [data-testid="stFileUploaderDropzone"] button:focus,
        [data-testid="baseButton-secondary"]:focus,
        [data-testid="baseButton-secondaryFormSubmit"]:focus {{
            color: white !important;
            border-color: #0b5ca0 !important;
            box-shadow: 0 0 0 0.2rem rgba(15, 108, 189, 0.18) !important;
        }}
        .block-container {{
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1320px;
        }}
        .hero-card {{
            background: linear-gradient(135deg, {panel} 0%, {accent_soft} 100%);
            border: 1px solid {border};
            border-radius: 28px;
            padding: 1.55rem 1.6rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
        }}
        .metric-card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 24px;
            padding: 1.15rem 1.15rem;
            min-height: 128px;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
        }}
        .metric-card.tone-green {{
            border-color: rgba(34, 197, 94, 0.35);
            box-shadow: 0 12px 24px rgba(34, 197, 94, 0.08);
        }}
        .metric-card.tone-amber {{
            border-color: rgba(245, 158, 11, 0.35);
            box-shadow: 0 12px 24px rgba(245, 158, 11, 0.08);
        }}
        .metric-card.tone-red {{
            border-color: rgba(239, 68, 68, 0.35);
            box-shadow: 0 12px 24px rgba(239, 68, 68, 0.08);
        }}
        .metric-value.tone-green {{
            color: #22c55e;
        }}
        .metric-value.tone-amber {{
            color: #f59e0b;
        }}
        .metric-value.tone-red {{
            color: #ef4444;
        }}
        .metric-status {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 0.45rem;
        }}
        .status-dot {{
            width: 0.95rem;
            height: 0.95rem;
            border-radius: 999px;
            display: inline-block;
            box-shadow: 0 0 0 4px rgba(255, 255, 255, 0.04);
        }}
        .status-dot.red {{
            background: #ef4444;
        }}
        .status-dot.amber {{
            background: #f59e0b;
        }}
        .status-dot.green {{
            background: #22c55e;
        }}
        .metric-label {{
            color: {muted};
            font-size: 0.92rem;
            margin-bottom: 0.35rem;
        }}
        .metric-value {{
            color: {text};
            font-size: 1.95rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }}
        .metric-pill {{
            display: inline-block;
            margin-top: 0.65rem;
            padding: 0.36rem 0.72rem;
            border-radius: 999px;
            background: {accent_soft};
            color: {accent};
            font-size: 0.88rem;
            font-weight: 600;
        }}
        .metric-pill.red {{
            background: {error_bg};
            color: {error_text};
        }}
        .metric-pill.amber {{
            background: {warning_bg};
            color: {warning_text};
        }}
        .metric-pill.green {{
            background: {success_bg};
            color: {success_text};
        }}
        .small-note {{
            color: {muted};
            font-size: 0.92rem;
        }}
        .section-shell {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 26px;
            padding: 1.2rem 1.3rem;
            margin: 0.35rem 0 1rem 0;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
        }}
        .section-title {{
            font-size: 1.08rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
            color: {text};
        }}
        .summary-box {{
            border-radius: 20px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.95rem;
            font-weight: 500;
            line-height: 1.5;
        }}
        .summary-success {{
            background: {success_bg};
            color: {success_text};
            border: 1px solid rgba(24,121,78,0.10);
        }}
        .summary-warning {{
            background: {warning_bg};
            color: {warning_text};
            border: 1px solid rgba(161,98,7,0.10);
        }}
        .summary-error {{
            background: {error_bg};
            color: {error_text};
            border: 1px solid rgba(180,35,24,0.10);
        }}
        .quick-card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 22px;
            padding: 1rem;
            min-height: 140px;
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.04);
        }}
        .quick-title {{
            font-weight: 700;
            margin-bottom: 0.55rem;
        }}
        .info-card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 24px;
            padding: 1.15rem 1.2rem;
            margin: 0.35rem 0 1rem 0;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.05);
        }}
        .info-title {{
            font-size: 1.05rem;
            font-weight: 750;
            margin-bottom: 0.85rem;
            color: {text};
        }}
        .info-list {{
            margin: 0;
            padding-left: 1.1rem;
        }}
        .info-list li {{
            margin-bottom: 0.45rem;
            color: {text};
        }}
        .records-table {{
            width: 100%;
            border-collapse: collapse;
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid {border};
            background: {panel};
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
            margin-bottom: 1rem;
        }}
        .records-table th {{
            text-align: left;
            padding: 0.85rem 1rem;
            background: {accent_soft};
            color: {text};
            border-bottom: 1px solid {border};
            font-weight: 700;
        }}
        .records-table td {{
            padding: 0.85rem 1rem;
            border-bottom: 1px solid {border};
            color: {text};
        }}
        .records-table tr:last-child td {{
            border-bottom: none;
        }}
        .preview-card {{
            background: {panel};
            border: 1px solid {border};
            border-radius: 22px;
            padding: 1rem 1.05rem;
            white-space: pre-wrap;
            line-height: 1.6;
            color: {text};
            box-shadow: 0 10px 22px rgba(15, 23, 42, 0.05);
        }}
        .report-section-title {{
            color: {text};
            font-weight: 800;
            margin: 0.8rem 0 0.35rem 0;
            font-size: 1rem;
        }}
        h1, h2, h3, h4, h5, h6, p, li, label {{
            color: {text} !important;
        }}
        [data-testid="stAlertContainer"] {{
            border-radius: 18px;
        }}
        div[data-baseweb="notification"] {{
            border-radius: 18px !important;
        }}
        div[data-baseweb="notification"][kind="success"] {{
            background: {success_bg} !important;
            color: {success_text} !important;
        }}
        div[data-baseweb="notification"][kind="warning"] {{
            background: {warning_bg} !important;
            color: {warning_text} !important;
        }}
        div[data-baseweb="notification"][kind="error"] {{
            background: {error_bg} !important;
            color: {error_text} !important;
        }}
        .stRadio label {{
            font-weight: 600;
        }}
        .stSelectbox label, .stFileUploader label {{
            font-weight: 700 !important;
        }}
        .stDataFrame {{
            border-radius: 18px;
            overflow: hidden;
        }}
        .stText, .stText * {{
            color: {text} !important;
        }}
        .stCodeBlock, pre, code {{
            color: {text} !important;
            background: {panel} !important;
        }}
        .stMarkdown table {{
            color: {text} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_data_file(filename: str) -> Path:
    for folder in DATA_DIR_CANDIDATES:
        path = folder / filename
        if path.exists():
            return path
    searched_locations = ", ".join(str(folder / filename) for folder in DATA_DIR_CANDIDATES)
    raise FileNotFoundError(f"Could not find {filename}. Looked in: {searched_locations}")


@st.cache_data
def load_data():
    patients = pd.read_csv(find_data_file("patients.csv"))
    conditions = pd.read_csv(find_data_file("conditions.csv"))
    encounters = pd.read_csv(find_data_file("encounters.csv"))

    try:
        observations = pd.read_csv(find_data_file("observations.csv.gz"), compression="gzip")
    except FileNotFoundError:
        observations = pd.read_csv(find_data_file("observations.csv"))

    patients = patients.copy()
    patients = patients[patients["STATE"].fillna("").str.lower() == TARGET_STATE.lower()].copy()
    observations = observations.copy()
    observations["PATIENT"] = observations["PATIENT"].astype(str)

    candidate_ids = set(patients["Id"].astype(str))
    patient_observations = observations[observations["PATIENT"].isin(candidate_ids)].copy()
    patient_observations["DESCRIPTION_LOWER"] = patient_observations["DESCRIPTION"].fillna("").str.lower()

    measure_patterns = {
        "has_glucose": ["glucose"],
        "has_bmi": ["body mass index", "bmi"],
        "has_creatinine": ["creatinine"],
        "has_systolic_bp": ["systolic blood pressure"],
    }

    coverage = pd.DataFrame({"Id": patients["Id"].astype(str)})
    for column_name, patterns in measure_patterns.items():
        matching_patients = set(
            patient_observations.loc[
                patient_observations["DESCRIPTION_LOWER"].apply(
                    lambda text: any(pattern in text for pattern in patterns)
                ),
                "PATIENT",
            ].astype(str)
        )
        coverage[column_name] = coverage["Id"].isin(matching_patients).astype(int)

    coverage["measure_coverage"] = coverage[
        ["has_glucose", "has_bmi", "has_creatinine", "has_systolic_bp"]
    ].sum(axis=1)
    coverage["is_default_demo_patient"] = (coverage["Id"] == DEFAULT_DEMO_PATIENT_ID).astype(int)

    patients = patients.merge(coverage, on="Id", how="left")
    for column_name in ["has_glucose", "has_bmi", "has_creatinine", "has_systolic_bp", "measure_coverage", "is_default_demo_patient"]:
        patients[column_name] = patients[column_name].fillna(0)

    patients = patients.sort_values(
        ["is_default_demo_patient", "measure_coverage", "has_glucose", "has_creatinine", "has_bmi", "has_systolic_bp", "Id"],
        ascending=[False, False, False, False, False, False, True],
    ).head(TARGET_PATIENT_COUNT)
    selected_patient_ids = set(patients["Id"].astype(str))

    observations = observations[observations["PATIENT"].astype(str).isin(selected_patient_ids)].copy()
    conditions = conditions[conditions["PATIENT"].astype(str).isin(selected_patient_ids)].copy()
    encounters = encounters[encounters["PATIENT"].astype(str).isin(selected_patient_ids)].copy()

    return patients, observations, conditions, encounters


@st.cache_data
def build_observation_features(observations: pd.DataFrame) -> pd.DataFrame:
    obs = observations.copy()
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs = obs.dropna(subset=["VALUE"])

    date_col = "DATE" if "DATE" in obs.columns else "START"
    obs[date_col] = pd.to_datetime(obs[date_col], errors="coerce")
    obs = obs.dropna(subset=[date_col])

    measures = {
        "glucose": ["glucose"],
        "bmi": ["body mass index", "bmi"],
        "creatinine": ["creatinine"],
        "systolic_bp": ["systolic blood pressure"],
    }

    features = None
    for feature_name, patterns in measures.items():
        mask = obs["DESCRIPTION"].fillna("").str.lower().apply(
            lambda text: any(pattern in text for pattern in patterns)
        )
        latest_rows = (
            obs.loc[mask, ["PATIENT", "VALUE", date_col]]
            .sort_values(date_col)
            .drop_duplicates("PATIENT", keep="last")
            .rename(columns={"VALUE": feature_name})
        )
        latest_rows = latest_rows[["PATIENT", feature_name]]
        features = latest_rows if features is None else features.merge(
            latest_rows,
            on="PATIENT",
            how="outer",
        )

    if features is None:
        return pd.DataFrame(columns=["PATIENT", "glucose", "bmi", "creatinine", "systolic_bp"])
    return features


@st.cache_data
def build_features(patients: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    patient_frame = patients.copy()
    patient_frame["Age"] = CURRENT_YEAR - pd.to_datetime(patient_frame["BIRTHDATE"]).dt.year
    patient_frame["GENDER"] = patient_frame["GENDER"].map({"M": 0, "F": 1}).fillna(0)

    obs_features = build_observation_features(observations)
    features = patient_frame[["Id", "Age", "GENDER"]].merge(
        obs_features,
        left_on="Id",
        right_on="PATIENT",
        how="left",
    )
    if "PATIENT" in features.columns:
        features = features.drop(columns=["PATIENT"])

    numeric_cols = ["Age", "GENDER", "glucose", "bmi", "creatinine", "systolic_bp"]
    for column in numeric_cols:
        if column not in features.columns:
            features[column] = 0.0
        features[column] = pd.to_numeric(features[column], errors="coerce")
        if column in {"glucose", "bmi", "creatinine", "systolic_bp"}:
            missing_flag_column = f"{column}_missing"
            features[missing_flag_column] = features[column].isna().astype(float)
        median_value = features[column].median()
        if pd.isna(median_value):
            median_value = 0.0
        features[column] = features[column].fillna(median_value)

    return features


@st.cache_data
def build_labels(patients: pd.DataFrame, conditions: pd.DataFrame) -> pd.DataFrame:
    labels = pd.DataFrame({"Id": patients["Id"]})
    condition_text = conditions["DESCRIPTION"].fillna("").str.lower()

    disease_patterns = {
        "diabetes": ["diabetes"],
        "kidney_disease": ["chronic kidney disease", "kidney disease", "renal"],
        "cardiovascular_disease": [
            "cardiovascular",
            "coronary",
            "heart disease",
            "myocardial",
            "cardiac",
            "stroke",
            "atherosclerosis",
        ],
        "hypertension": ["hypertension", "high blood pressure"],
    }

    for disease_name, patterns in disease_patterns.items():
        mask = condition_text.apply(lambda text: any(pattern in text for pattern in patterns))
        labels[disease_name] = patients["Id"].isin(
            conditions.loc[mask, "PATIENT"]
        ).astype(int)

    return labels


@st.cache_resource
def train_model(features: pd.DataFrame, labels: pd.DataFrame):
    x_data = features.drop(columns=["Id"])
    disease_models = {}

    for disease_name in [col for col in labels.columns if col != "Id"]:
        y_data = labels[disease_name]

        if y_data.nunique() < 2:
            disease_models[disease_name] = None
            continue

        x_train, x_test, y_train, y_test = train_test_split(
            x_data,
            y_data,
            test_size=0.2,
            random_state=42,
            stratify=y_data,
        )

        positive_count = int(y_train.sum())
        negative_count = int((y_train == 0).sum())
        scale_pos_weight = negative_count / max(positive_count, 1)

        model = xgb.XGBClassifier(
            eval_metric="logloss",
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=1,
        )
        model.fit(x_train, y_train)
        disease_models[disease_name] = model

    return disease_models


def get_patient_latest_values(patient_id: str, observations: pd.DataFrame) -> dict:
    obs = observations.copy()
    obs = obs[obs["PATIENT"] == patient_id]
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs = obs.dropna(subset=["VALUE"])

    date_col = "DATE" if "DATE" in obs.columns else "START"
    obs[date_col] = pd.to_datetime(obs[date_col], errors="coerce")
    obs = obs.dropna(subset=[date_col])

    result = {}
    targets = {
        "Blood sugar": ["glucose"],
        "BMI": ["body mass index", "bmi"],
        "Creatinine": ["creatinine"],
        "Systolic blood pressure": ["systolic blood pressure"],
    }

    for label, patterns in targets.items():
        subset = obs[
            obs["DESCRIPTION"].fillna("").str.lower().apply(
                lambda text: any(pattern in text for pattern in patterns)
            )
        ].sort_values(date_col)
        if not subset.empty:
            result[label] = float(subset.iloc[-1]["VALUE"])

    return result


@st.cache_data
def build_latest_values_lookup(observations: pd.DataFrame) -> dict:
    obs = observations.copy()
    obs["PATIENT"] = obs["PATIENT"].astype(str)
    obs["VALUE"] = pd.to_numeric(obs["VALUE"], errors="coerce")
    obs = obs.dropna(subset=["VALUE"])

    date_col = "DATE" if "DATE" in obs.columns else "START"
    obs[date_col] = pd.to_datetime(obs[date_col], errors="coerce")
    obs = obs.dropna(subset=[date_col])
    obs["DESCRIPTION_LOWER"] = obs["DESCRIPTION"].fillna("").str.lower()

    targets = {
        "Blood sugar": ["glucose"],
        "BMI": ["body mass index", "bmi"],
        "Creatinine": ["creatinine"],
        "Systolic blood pressure": ["systolic blood pressure"],
    }

    latest_lookup: dict[str, dict] = {}

    for label, patterns in targets.items():
        subset = obs[
            obs["DESCRIPTION_LOWER"].apply(lambda text: any(pattern in text for pattern in patterns))
        ][["PATIENT", "VALUE", date_col]].copy()

        if subset.empty:
            continue

        subset = subset.sort_values(date_col).drop_duplicates("PATIENT", keep="last")
        for _, row in subset.iterrows():
            patient_id = row["PATIENT"]
            if patient_id not in latest_lookup:
                latest_lookup[patient_id] = {}
            latest_lookup[patient_id][label] = float(row["VALUE"])

    return latest_lookup


@st.cache_data
def build_patient_history_lookups(conditions: pd.DataFrame, encounters: pd.DataFrame) -> tuple[dict, dict]:
    condition_lookup = {
        str(patient_id): group.copy()
        for patient_id, group in conditions.groupby("PATIENT", sort=False)
    }
    encounter_lookup = {
        str(patient_id): group.copy()
        for patient_id, group in encounters.groupby("PATIENT", sort=False)
    }
    return condition_lookup, encounter_lookup


def interpret_measure(label: str, value: float) -> str:
    if label == "Blood sugar":
        if value >= 126:
            return "higher than the usual target range"
        if value >= 100:
            return "slightly above the usual target range"
        return "within the usual target range"
    if label == "BMI":
        if value >= 30:
            return "in a range associated with higher health risk"
        if value >= 25:
            return "above the recommended range"
        return "within the recommended range"
    if label == "Creatinine":
        if value > 1.3:
            return "higher than expected and may need follow-up"
        return "within the usual target range"
    if label == "Systolic blood pressure":
        if value >= 140:
            return "high and may need medical follow-up"
        if value >= 120:
            return "a little above the ideal range"
        return "within the usual target range"
    return "available in your record"


def build_record_summary(latest_values: dict) -> list[str]:
    if not latest_values:
        return ["We could not find enough recent lab or vital data for a detailed summary."]
    return [f"{label}: {value:.1f} ({interpret_measure(label, value)})" for label, value in latest_values.items()]


def build_next_steps(probability: float, latest_values: dict) -> list[str]:
    steps = []
    if probability >= 0.6:
        steps.append("Schedule a follow-up visit to review your recent lab results.")
    elif probability >= 0.35:
        steps.append("Plan a routine check-up and ask whether repeat lab testing would help.")
    else:
        steps.append("Keep up your current routine and continue regular preventive care.")

    if latest_values.get("Blood sugar", 0) >= 100:
        steps.append("Ask whether repeat blood sugar or HbA1c testing is needed.")
    if latest_values.get("Systolic blood pressure", 0) >= 120:
        steps.append("Track blood pressure over time because trends matter more than one reading.")
    if latest_values.get("Creatinine", 0) > 1.3:
        steps.append("Discuss kidney function follow-up and whether repeat testing is needed.")
    if latest_values.get("BMI", 0) >= 25:
        steps.append("Focus on small daily changes like walking, sleep, and balanced meals.")
    return steps


def build_risk_reasons(patient_row: pd.Series, latest_values: dict, probability: float) -> list[str]:
    reasons = []
    age_value = CURRENT_YEAR - pd.to_datetime(patient_row["BIRTHDATE"]).year

    if age_value >= 45:
        reasons.append("Age can increase long-term chronic disease risk.")
    if latest_values.get("Blood sugar", 0) >= 100:
        reasons.append("Recent blood sugar values are above the ideal range.")
    if latest_values.get("BMI", 0) >= 25:
        reasons.append("BMI is above the recommended range.")
    if latest_values.get("Systolic blood pressure", 0) >= 120:
        reasons.append("Blood pressure is above the ideal range.")
    if latest_values.get("Creatinine", 0) > 1.3:
        reasons.append("Creatinine is elevated, so kidney follow-up may be important.")

    if not reasons:
        reasons.append(
            "The available record does not show many strong warning signs right now."
            if probability < 0.35
            else "The record shows a pattern that may be worth discussing during a follow-up visit."
        )
    return reasons


def get_disease_probabilities(models: dict, patient_features: pd.DataFrame) -> dict:
    probabilities = {}
    for disease_name, model in models.items():
        if model is None:
            probabilities[disease_name] = 0.0
        else:
            probabilities[disease_name] = float(model.predict_proba(patient_features)[0][1])
    return probabilities


def combine_uploaded_report_values(uploaded_reports: list[dict] | None) -> dict:
    combined_values: dict[str, float] = {}
    for report in uploaded_reports or []:
        if report.get("values"):
            combined_values.update(report["values"])
    return combined_values


def apply_uploaded_values_to_features(patient_features: pd.DataFrame, uploaded_reports: list[dict] | None) -> pd.DataFrame:
    adjusted_features = patient_features.copy()
    uploaded_values = combine_uploaded_report_values(uploaded_reports)
    feature_map = {
        "Blood sugar": "glucose",
        "BMI": "bmi",
        "Creatinine": "creatinine",
        "Systolic blood pressure": "systolic_bp",
    }

    for label, feature_name in feature_map.items():
        if label in uploaded_values:
            adjusted_features[feature_name] = float(uploaded_values[label])
            missing_flag_column = f"{feature_name}_missing"
            if missing_flag_column in adjusted_features.columns:
                adjusted_features[missing_flag_column] = 0.0

    return adjusted_features


def validate_uploaded_file(uploaded_file, patient_id: str):
    raw_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
    file_suffix = Path(file_name.lower()).suffix.lower()

    parsed_upload = {
        "file_name": file_name,
        "file_type": file_suffix or "unknown",
        "patient_id": None,
        "values": {},
        "notes": [],
        "uploaded_at": pd.Timestamp.now().isoformat(),
    }

    if file_suffix == ".csv":
        uploaded_frame = pd.read_csv(BytesIO(raw_bytes))
    elif file_suffix == ".json":
        uploaded_frame = pd.read_json(BytesIO(raw_bytes))
    elif file_suffix in [".txt", ".text"]:
        uploaded_text = raw_bytes.decode("utf-8", errors="ignore")
        parsed_upload = parse_hospital_text_report(uploaded_text, file_name)

        if parsed_upload["patient_id"] and parsed_upload["patient_id"] != str(patient_id):
            return "error", "This uploaded report does not belong to the selected patient.", None
        if parsed_upload["patient_id"] is None:
            return "warning", "This text report uploaded, but no patient ID could be verified.", parsed_upload
        return "success", f"{file_name} uploaded and matched to the selected patient.", parsed_upload
    else:
        return "success", f"{file_name} uploaded successfully.", None

    candidate_columns = ["PATIENT", "patient_id", "Patient", "Id", "ID"]
    patient_column = next((col for col in candidate_columns if col in uploaded_frame.columns), None)

    if patient_column is None:
        return "warning", "This file uploaded, but it does not include a patient ID column for verification.", None

    patient_values = uploaded_frame[patient_column].astype(str).str.strip()
    match_count = int((patient_values == str(patient_id)).sum())

    if match_count > 0:
        parsed_upload["patient_id"] = str(patient_id)
        parsed_upload["values"] = extract_measurements_from_dataframe(uploaded_frame)

    if match_count == 0:
        return "error", "This file does not belong to the patient you selected.", None
    if match_count < len(uploaded_frame):
        return "info", f"{file_name} uploaded. Matching rows found for this patient: {match_count}.", parsed_upload
    return "success", f"{file_name} matches the selected patient.", parsed_upload


def extract_measurements_from_dataframe(uploaded_frame: pd.DataFrame) -> dict:
    extracted = {}
    column_map = {str(col).strip().lower(): col for col in uploaded_frame.columns}

    measurement_patterns = {
        "Blood sugar": ["glucose", "blood sugar"],
        "Creatinine": ["creatinine"],
        "BMI": ["bmi", "body mass index"],
        "Systolic blood pressure": ["systolic blood pressure", "systolic_bp", "sbp"],
    }

    for label, patterns in measurement_patterns.items():
        matched_column = next(
            (column_map[key] for key in column_map if any(pattern in key for pattern in patterns)),
            None,
        )
        if matched_column is not None:
            series = pd.to_numeric(uploaded_frame[matched_column], errors="coerce").dropna()
            if not series.empty:
                extracted[label] = float(series.iloc[0])

    if extracted:
        return extracted

    name_column = next(
        (
            column_map[key]
            for key in column_map
            if key in {"test_name", "test", "analyte", "measurement", "observation", "result_name"}
        ),
        None,
    )
    value_column = next(
        (
            column_map[key]
            for key in column_map
            if key in {"value", "result_value", "result", "numeric_result", "observation_value"}
        ),
        None,
    )

    if name_column is not None and value_column is not None:
        name_series = uploaded_frame[name_column].astype(str).str.strip().str.lower()
        value_series = pd.to_numeric(uploaded_frame[value_column], errors="coerce")

        for label, patterns in measurement_patterns.items():
            mask = name_series.apply(lambda text: any(pattern in text for pattern in patterns))
            matches = value_series[mask].dropna()
            if not matches.empty:
                extracted[label] = float(matches.iloc[0])

    return extracted


def parse_hospital_text_report(uploaded_text: str, file_name: str) -> dict:
    parsed = {
        "file_name": file_name,
        "file_type": ".txt",
        "patient_id": None,
        "values": {},
        "notes": [],
    }

    patient_match = re.search(r"Patient ID:\s*([A-Za-z0-9\-]+)", uploaded_text, flags=re.IGNORECASE)
    if patient_match:
        parsed["patient_id"] = patient_match.group(1).strip()

    pattern_map = {
        "Blood sugar": r"(Glucose|Blood Sugar):\s*([0-9]+(?:\.[0-9]+)?)",
        "Creatinine": r"(Creatinine):\s*([0-9]+(?:\.[0-9]+)?)",
        "BMI": r"(BMI|Body Mass Index):\s*([0-9]+(?:\.[0-9]+)?)",
        "Systolic blood pressure": r"(Systolic Blood Pressure|SBP):\s*([0-9]+(?:\.[0-9]+)?)",
    }

    for label, pattern in pattern_map.items():
        match = re.search(pattern, uploaded_text, flags=re.IGNORECASE)
        if match:
            parsed["values"][label] = float(match.group(2))

    note_match = re.search(r"Clinical Note:\s*(.+)", uploaded_text, flags=re.IGNORECASE)
    if note_match:
        parsed["notes"].append(note_match.group(1).strip())

    return parsed


def build_uploaded_report_summary(uploaded_report: dict | None) -> str:
    if not uploaded_report:
        return "No uploaded hospital-style report has been saved in this session yet."

    lines = [
        f"Uploaded file: {uploaded_report.get('file_name', 'Unknown file')}",
        f"Detected file type: {uploaded_report.get('file_type', 'Unknown')}",
    ]

    if uploaded_report.get("patient_id"):
        lines.append(f"Verified patient ID: {uploaded_report['patient_id']}")

    values = uploaded_report.get("values", {})
    if values:
        lines.append("")
        lines.append("Extracted values")
        for label, value in values.items():
            lines.append(f"- {label}: {value}")

    notes = uploaded_report.get("notes", [])
    if notes:
        lines.append("")
        lines.append("Clinical note")
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


def build_uploaded_reports_summary(uploaded_reports: list[dict] | None) -> str:
    if not uploaded_reports:
        return "No uploaded hospital-style reports have been saved in this session yet."

    lines = [f"Uploaded files in this session: {len(uploaded_reports)}"]
    combined_values = combine_uploaded_report_values(uploaded_reports)

    if combined_values:
        lines.append("")
        lines.append("Active values currently applied to the dashboard")
        for label, value in combined_values.items():
            lines.append(f"- {label}: {value}")

    for index, report in enumerate(uploaded_reports, start=1):
        lines.append("")
        lines.append(f"File {index}")
        lines.append(build_uploaded_report_summary(report))

    return "\n".join(lines)


def risk_level(probability: float) -> str:
    if probability >= 0.6:
        return "High"
    if probability >= 0.35:
        return "Moderate"
    return "Low"


def pretty_disease_name(disease_name: str) -> str:
    names = {
        "diabetes": "Diabetes",
        "kidney_disease": "Kidney Disease",
        "cardiovascular_disease": "Cardiovascular Disease",
        "hypertension": "Hypertension",
    }
    return names.get(disease_name, disease_name.replace("_", " ").title())


def overall_alert_level(disease_probabilities: dict) -> str:
    if not disease_probabilities:
        return "Low"
    highest_probability = max(disease_probabilities.values())
    return risk_level(highest_probability)


def kidney_status(latest_values: dict) -> str:
    creatinine = latest_values.get("Creatinine")
    if creatinine is None:
        return "No recent value"
    if creatinine > 1.3:
        return "Needs review"
    return "Stable"


def bp_status(latest_values: dict) -> str:
    systolic = latest_values.get("Systolic blood pressure")
    if systolic is None:
        return "No recent value"
    if systolic >= 140:
        return "High"
    if systolic >= 120:
        return "Elevated"
    return "Normal"


def tone_from_risk(probability: float) -> str:
    if probability >= 0.6:
        return "red"
    if probability >= 0.35:
        return "amber"
    return "green"


def tone_from_score(score: int) -> str:
    if score < 40:
        return "red"
    if score < 70:
        return "amber"
    return "green"


def tone_from_bp(latest_values: dict) -> str:
    status = bp_status(latest_values)
    if status == "High":
        return "red"
    if status in {"Elevated", "No recent value"}:
        return "amber"
    return "green"


def icon_for_tone(tone: str) -> str:
    return {"red": "&uarr;", "amber": "!", "green": "&check;"}.get(tone, "&bull;")


def get_patient_history(patient_id: str, conditions_df: pd.DataFrame, encounters_df: pd.DataFrame):
    patient_conditions = conditions_df[conditions_df["PATIENT"] == patient_id].copy()
    patient_encounters = encounters_df[encounters_df["PATIENT"] == patient_id].copy()
    return patient_conditions, patient_encounters


def has_condition_history(patient_conditions: pd.DataFrame, keywords: list[str]) -> bool:
    if patient_conditions.empty:
        return False
    condition_text = patient_conditions["DESCRIPTION"].fillna("").str.lower()
    return bool(condition_text.apply(lambda text: any(keyword in text for keyword in keywords)).any())


def condition_history_boost(patient_conditions: pd.DataFrame, keywords: list[str]) -> float:
    if patient_conditions.empty:
        return 0.0

    patient_conditions = patient_conditions.copy()
    patient_conditions["DESCRIPTION_LOWER"] = patient_conditions["DESCRIPTION"].fillna("").str.lower()
    matching_conditions = patient_conditions[
        patient_conditions["DESCRIPTION_LOWER"].apply(lambda text: any(keyword in text for keyword in keywords))
    ].copy()

    if matching_conditions.empty:
        return 0.0

    if "START" in matching_conditions.columns:
        matching_conditions["START"] = pd.to_datetime(matching_conditions["START"], errors="coerce")
        latest_start = matching_conditions["START"].max()
        if pd.notna(latest_start):
            days_since_latest = (pd.Timestamp.now().normalize() - latest_start.normalize()).days
            if days_since_latest <= 180:
                return 0.55
            if days_since_latest <= 365:
                return 0.45

    if len(matching_conditions) >= 2:
        return 0.40
    return 0.30


def select_clinically_useful_conditions(patient_conditions: pd.DataFrame) -> pd.DataFrame:
    if patient_conditions.empty:
        return patient_conditions

    condition_text = patient_conditions["DESCRIPTION"].fillna("").str.lower()
    include_keywords = [
        "diabetes",
        "hypertension",
        "kidney",
        "renal",
        "cardio",
        "heart",
        "coronary",
        "stroke",
        "cholesterol",
        "hyperlip",
        "obesity",
        "prediabetes",
        "blood pressure",
        "disease",
        "chronic",
        "ckd",
    ]
    exclude_keywords = [
        "medication review due",
        "not in labor force",
        "part-time employment",
        "full-time employment",
        "social isolation",
        "stress",
        "drinking behavior",
        "finding",
        "sinusitis",
        "tooth",
        "teeth",
        "gingivitis",
        "jaw",
        "alveolitis",
        "palatinus",
        "viral",
    ]

    include_mask = condition_text.apply(lambda text: any(keyword in text for keyword in include_keywords))
    exclude_mask = condition_text.apply(lambda text: any(keyword in text for keyword in exclude_keywords))
    filtered = patient_conditions[include_mask & ~exclude_mask].copy()

    return filtered.sort_values("START", ascending=False)


def build_record_based_disease_scores(
    latest_values: dict,
    patient_conditions: pd.DataFrame,
    model_probabilities: dict,
    patient_row: pd.Series,
    patient_features: pd.DataFrame | None = None,
) -> dict:
    scores = {name: float(model_probabilities.get(name, 0.0)) for name in [
        "kidney_disease",
        "diabetes",
        "cardiovascular_disease",
        "hypertension",
    ]}

    age_value = CURRENT_YEAR - pd.to_datetime(patient_row["BIRTHDATE"]).year if pd.notna(patient_row["BIRTHDATE"]) else 0
    systolic_bp = latest_values.get("Systolic blood pressure")
    blood_sugar = latest_values.get("Blood sugar")
    creatinine = latest_values.get("Creatinine")
    bmi = latest_values.get("BMI")

    diabetes_history_boost = condition_history_boost(patient_conditions, ["diabetes", "prediabetes"])
    kidney_history_boost = condition_history_boost(patient_conditions, ["kidney", "renal", "chronic kidney disease", "ckd"])
    hypertension_history_boost = condition_history_boost(patient_conditions, ["hypertension", "high blood pressure"])
    cardiovascular_history_boost = condition_history_boost(patient_conditions, ["cardiovascular", "coronary", "heart disease", "myocardial", "cardiac", "stroke", "atherosclerosis"])

    if diabetes_history_boost:
        scores["diabetes"] = max(scores["diabetes"], diabetes_history_boost)
    if kidney_history_boost:
        scores["kidney_disease"] = max(scores["kidney_disease"], kidney_history_boost)
    if hypertension_history_boost:
        scores["hypertension"] = max(scores["hypertension"], hypertension_history_boost)
    if cardiovascular_history_boost:
        scores["cardiovascular_disease"] = max(scores["cardiovascular_disease"], cardiovascular_history_boost)

    if blood_sugar is not None:
        if blood_sugar >= 200:
            scores["diabetes"] = max(scores["diabetes"], 0.95)
        elif blood_sugar >= 126:
            scores["diabetes"] = max(scores["diabetes"], 0.80)
        elif blood_sugar >= 100:
            scores["diabetes"] = max(scores["diabetes"], 0.45)

    if creatinine is not None:
        if creatinine >= 2.0:
            scores["kidney_disease"] = max(scores["kidney_disease"], 0.95)
        elif creatinine > 1.3:
            scores["kidney_disease"] = max(scores["kidney_disease"], 0.80)
        elif creatinine > 1.1:
            scores["kidney_disease"] = max(scores["kidney_disease"], 0.45)

    if systolic_bp is not None:
        if systolic_bp >= 160:
            scores["hypertension"] = max(scores["hypertension"], 0.95)
            scores["cardiovascular_disease"] = max(scores["cardiovascular_disease"], 0.75)
        elif systolic_bp >= 140:
            scores["hypertension"] = max(scores["hypertension"], 0.85)
            scores["cardiovascular_disease"] = max(scores["cardiovascular_disease"], 0.60)
        elif systolic_bp >= 130:
            scores["hypertension"] = max(scores["hypertension"], 0.65)
            scores["cardiovascular_disease"] = max(scores["cardiovascular_disease"], 0.45)
        elif systolic_bp >= 120:
            scores["hypertension"] = max(scores["hypertension"], 0.35)
            scores["cardiovascular_disease"] = max(scores["cardiovascular_disease"], 0.25)

    if bmi is not None:
        if bmi >= 35:
            scores["diabetes"] = max(scores["diabetes"], 0.55)
            scores["cardiovascular_disease"] = max(scores["cardiovascular_disease"], 0.45)
        elif bmi >= 30:
            scores["diabetes"] = max(scores["diabetes"], 0.45)
            scores["cardiovascular_disease"] = max(scores["cardiovascular_disease"], 0.35)
        elif bmi >= 25:
            scores["diabetes"] = max(scores["diabetes"], 0.30)
            scores["cardiovascular_disease"] = max(scores["cardiovascular_disease"], 0.25)

    if age_value >= 55:
        scores["cardiovascular_disease"] = max(scores["cardiovascular_disease"], 0.30)
    elif age_value >= 45:
        scores["cardiovascular_disease"] = max(scores["cardiovascular_disease"], 0.20)

    if patient_features is not None and not patient_features.empty:
        feature_row = patient_features.iloc[0]
        missing_feature_map = {
            "diabetes": ["glucose_missing", "bmi_missing"],
            "kidney_disease": ["creatinine_missing"],
            "cardiovascular_disease": ["systolic_bp_missing", "bmi_missing"],
            "hypertension": ["systolic_bp_missing"],
        }
        for disease_name, missing_columns in missing_feature_map.items():
            missing_values = [float(feature_row.get(column, 0.0)) for column in missing_columns if column in feature_row.index]
            if missing_values:
                missing_fraction = sum(missing_values) / len(missing_values)
                if missing_fraction > 0:
                    scores[disease_name] = scores[disease_name] * (1 - (0.35 * missing_fraction))

    return {name: max(0.0, min(1.0, value)) for name, value in scores.items()}


def calculate_overall_health_score(display_scores: dict) -> int:
    if not display_scores:
        return 50

    values = list(display_scores.values())
    highest_score = max(values)
    average_score = sum(values) / len(values)
    risk_component = (0.65 * highest_score) + (0.35 * average_score)
    return max(0, min(100, int(round((1 - risk_component) * 100))))


def build_clinical_flags(latest_values: dict, disease_probabilities: dict) -> list[str]:
    flags = []

    systolic_bp = latest_values.get("Systolic blood pressure")
    blood_sugar = latest_values.get("Blood sugar")
    creatinine = latest_values.get("Creatinine")
    bmi = latest_values.get("BMI")

    if systolic_bp is not None and systolic_bp >= 120:
        flags.append(f"Systolic blood pressure: {systolic_bp:.0f} mmHg")
    if blood_sugar is not None and blood_sugar >= 100:
        flags.append(f"Blood sugar: {blood_sugar:.1f}")
    if creatinine is not None and creatinine > 1.3:
        flags.append(f"Creatinine: {creatinine:.2f}")
    if bmi is not None and bmi >= 30:
        flags.append(f"BMI: {bmi:.1f}")

    for disease_name in ["kidney_disease", "diabetes", "cardiovascular_disease", "hypertension"]:
        disease_probability = disease_probabilities.get(disease_name, 0.0)
        if disease_probability >= 0.2:
            flags.append(
                f"{pretty_disease_name(disease_name)} risk estimate: {risk_level(disease_probability)} ({disease_probability * 100:.1f}%)"
            )

    if not flags:
        flags.append("No major flagged value was found in the current record snapshot.")

    return flags[:6]


def build_doctor_key_findings(latest_values: dict, disease_probabilities: dict) -> list[str]:
    findings = []

    systolic_bp = latest_values.get("Systolic blood pressure")
    blood_sugar = latest_values.get("Blood sugar")
    creatinine = latest_values.get("Creatinine")
    bmi = latest_values.get("BMI")

    if systolic_bp is not None and systolic_bp >= 120:
        findings.append(f"Elevated systolic blood pressure: {systolic_bp:.0f} mmHg.")
    if blood_sugar is not None and blood_sugar >= 100:
        findings.append(f"Blood sugar above ideal range: {blood_sugar:.1f}.")
    if creatinine is not None and creatinine > 1.3:
        findings.append(f"Creatinine above reference threshold: {creatinine:.2f}.")
    if bmi is not None and bmi >= 25:
        findings.append(f"BMI above recommended range: {bmi:.1f}.")

    ranked_risks = sorted(disease_probabilities.items(), key=lambda item: item[1], reverse=True)
    for disease_name, probability in ranked_risks[:2]:
        if probability >= 0.2:
            findings.append(
                f"{pretty_disease_name(disease_name)} risk signal in the {risk_level(probability).lower()} range ({probability * 100:.1f}%)."
            )

    if not findings:
        findings.append("No clearly abnormal value was identified in the current record snapshot.")

    return findings[:5]


def build_doctor_follow_up(latest_values: dict, disease_probabilities: dict) -> list[str]:
    follow_up = []

    if latest_values.get("Systolic blood pressure", 0) >= 120:
        follow_up.append("Consider repeat blood pressure measurement or home BP trend review.")
    if latest_values.get("Blood sugar", 0) >= 100:
        follow_up.append("Consider repeat glucose or HbA1c testing if clinically appropriate.")
    if latest_values.get("Creatinine", 0) > 1.3:
        follow_up.append("Consider kidney function follow-up and repeat renal labs if indicated.")

    highest_disease = max(disease_probabilities, key=disease_probabilities.get)
    highest_probability = disease_probabilities.get(highest_disease, 0.0)
    if highest_probability >= 0.35:
        follow_up.append(f"Review {pretty_disease_name(highest_disease).lower()} risk signal in context of prior history and current labs.")

    if not follow_up:
        follow_up.append("Continue routine preventive follow-up and trend monitoring.")

    return follow_up[:4]


def build_report_text(
    patient_name: str,
    patient_row: pd.Series,
    latest_values: dict,
    disease_probabilities: dict,
    reasons: list[str],
    next_steps: list[str],
    patient_conditions: pd.DataFrame,
    patient_encounters: pd.DataFrame,
) -> str:
    useful_conditions = select_clinically_useful_conditions(patient_conditions)
    clinical_flags = build_clinical_flags(latest_values, disease_probabilities)
    age_value = CURRENT_YEAR - pd.to_datetime(patient_row["BIRTHDATE"]).year if pd.notna(patient_row["BIRTHDATE"]) else "Not available"
    sex_value = patient_row["GENDER"].title() if pd.notna(patient_row["GENDER"]) else "Not available"

    lines = [
        "Patient summary:",
        f"- Age: {age_value}",
        f"- Sex: {sex_value}",
        "",
        "Risk checks:",
    ]

    for disease_name in ["kidney_disease", "diabetes", "cardiovascular_disease", "hypertension"]:
        disease_probability = disease_probabilities.get(disease_name, 0.0)
        lines.append(
            f"- {pretty_disease_name(disease_name)}: {risk_level(disease_probability)} ({disease_probability * 100:.1f}%)"
        )

    lines.append("")
    lines.append("Latest values found in the record:")
    for summary in build_record_summary(latest_values):
        lines.append(f"- {summary}")

    lines.append("")
    lines.append("Key flagged findings:")
    for item in clinical_flags:
        lines.append(f"- {item}")

    lines.append("")
    lines.append("Clinically useful history to share:")
    if useful_conditions.empty:
        lines.append("- No chronic-condition history was found in the current record.")
    else:
        recent_conditions = (
            useful_conditions[["START", "DESCRIPTION"]]
            .copy()
            .sort_values("START", ascending=False)
            .head(8)
        )
        for _, row in recent_conditions.iterrows():
            lines.append(f"- {row['START']}: {row['DESCRIPTION']}")

    lines.append("")
    lines.append("Recent doctor visits and care events:")
    if patient_encounters.empty:
        lines.append("- No recent visits found.")
    else:
        recent_visits = (
            patient_encounters[["START", "DESCRIPTION"]]
            .copy()
            .sort_values("START", ascending=False)
            .head(10)
        )
        for _, row in recent_visits.iterrows():
            lines.append(f"- {row['START']}: {row['DESCRIPTION']}")
    return "\n".join(lines)


def download_bytes(text: str) -> bytes:
    buffer = BytesIO()
    buffer.write(text.encode("utf-8"))
    return buffer.getvalue()


def render_html_table(dataframe: pd.DataFrame):
    st.markdown(
        dataframe.to_html(index=False, classes="records-table", escape=False, border=0),
        unsafe_allow_html=True,
    )


def report_preview_html(report_text: str) -> str:
    escaped = (
        report_text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n\n", "<br><br>")
        .replace("\n", "<br>")
    )
    return f'<div class="preview-card">{escaped}</div>'


TERM_EXPLANATIONS = {
    "bmi": "BMI stands for body mass index. It is a simple estimate that compares your weight with your height. It can help show whether your weight is in a lower, usual, or higher range.",
    "body mass index": "Body mass index, or BMI, is a simple estimate that compares your weight with your height. It can help show whether your weight is in a lower, usual, or higher range.",
    "creatinine": "Creatinine is a blood test often used to check how your kidneys are working. A higher result can mean your kidneys may need closer follow-up.",
    "egfr": "eGFR is an estimate of how well your kidneys are filtering blood. Lower values can suggest reduced kidney function.",
    "hba1c": "HbA1c is a blood test that shows your average blood sugar level over the past two to three months.",
    "glucose": "Glucose means blood sugar. It shows how much sugar is in your blood at the time of the test.",
    "blood sugar": "Blood sugar shows how much sugar is in your blood at the time of the test. Higher values may need repeat testing or follow-up.",
    "hypertension": "Hypertension is another word for high blood pressure. It means the pressure in your blood vessels is staying higher than ideal.",
    "systolic blood pressure": "Systolic blood pressure is the top number in a blood pressure reading. It shows how strongly blood pushes against your arteries when the heart beats.",
    "kidney disease": "Kidney disease means the kidneys are not working as well as expected. It can be mild at first and may need lab follow-up over time.",
    "cardiovascular disease": "Cardiovascular disease is a broad term for health problems involving the heart or blood vessels.",
    "diabetes": "Diabetes is a condition where the body has trouble controlling blood sugar. Doctors often use blood sugar and HbA1c tests to check it.",
    "cholesterol": "Cholesterol is a type of fat in the blood. Some cholesterol is needed, but higher levels may raise heart risk over time.",
    "risk": "Risk means chance, not certainty. A higher risk result does not confirm disease, but it does suggest an area that may need more follow-up.",
    "risk score": "A risk score is an estimate of how likely a health problem may be. It helps point out areas to watch, but it does not confirm a diagnosis.",
    "health score": "The health score is a simple app summary based on the current record. It is meant to be easy to understand, not a medical diagnosis by itself.",
    "trend": "A trend means how a result changes over time. Trends are important because one value alone may not show the full picture.",
    "lab result": "A lab result is one test value from your record. Doctors usually interpret it together with your history, symptoms, and repeat testing if needed.",
    "elevated": "Elevated means above the ideal or expected range, but not always dangerously high. It usually means the value deserves some attention or follow-up.",
}


def explain_medical_term(term: str) -> str:
    normalized_term = term.strip().lower()
    if not normalized_term:
        return "Type a medical word or test name and the app will explain it in simple language."

    for known_term, explanation in TERM_EXPLANATIONS.items():
        if normalized_term == known_term or normalized_term in known_term or known_term in normalized_term:
            return explanation

    return (
        f'"{term.strip()}" is not in the built-in glossary yet. A good way to ask a doctor is: '
        f'"Can you explain what {term.strip()} means in simple words and why it matters for me?"'
    )


def build_personal_focus(latest_values: dict, disease_probabilities: dict) -> list[str]:
    focus_points = []

    systolic_bp = latest_values.get("Systolic blood pressure")
    blood_sugar = latest_values.get("Blood sugar")
    creatinine = latest_values.get("Creatinine")
    bmi = latest_values.get("BMI")

    if systolic_bp is not None:
        if systolic_bp >= 140:
            focus_points.append(f"Your latest systolic blood pressure is {systolic_bp:.0f} mmHg, which is clearly above the ideal range.")
        elif systolic_bp >= 120:
            focus_points.append(f"Your latest systolic blood pressure is {systolic_bp:.0f} mmHg, which is a little above the ideal range.")

    if blood_sugar is not None:
        if blood_sugar >= 126:
            focus_points.append(f"Your latest blood sugar is {blood_sugar:.1f}, which is above the usual target range.")
        elif blood_sugar >= 100:
            focus_points.append(f"Your latest blood sugar is {blood_sugar:.1f}, which is slightly above the usual target range.")

    if creatinine is not None and creatinine > 1.3:
        focus_points.append(f"Your latest creatinine is {creatinine:.2f}, so kidney follow-up may matter here.")

    if bmi is not None:
        if bmi >= 30:
            focus_points.append(f"Your BMI is {bmi:.1f}, which is in a higher-risk range.")
        elif bmi >= 25:
            focus_points.append(f"Your BMI is {bmi:.1f}, which is above the recommended range.")

    highest_disease = max(disease_probabilities, key=disease_probabilities.get)
    highest_probability = disease_probabilities.get(highest_disease, 0.0)
    if highest_probability >= 0.35:
        focus_points.append(
            f"Your highest current risk score is for {pretty_disease_name(highest_disease).lower()} at {highest_probability * 100:.1f}%."
        )

    if not focus_points:
        focus_points.append("Your current record does not show a strong warning signal in the values available here.")

    return focus_points[:4]


def build_questions_for_doctor(latest_values: dict, disease_probabilities: dict, patient_conditions: pd.DataFrame) -> list[str]:
    questions = []

    if latest_values.get("Systolic blood pressure", 0) >= 120:
        questions.append("Is my blood pressure high enough that I should start checking it regularly at home?")
    if latest_values.get("Blood sugar", 0) >= 100:
        questions.append("Do my blood sugar results suggest I need repeat testing or an HbA1c test?")
    if latest_values.get("Creatinine", 0) > 1.3:
        questions.append("Should I repeat kidney function testing based on my recent creatinine result?")
    if latest_values.get("BMI", 0) >= 25:
        questions.append("Would weight, food, activity, or sleep changes make a meaningful difference for my results?")

    highest_disease = max(disease_probabilities, key=disease_probabilities.get)
    highest_probability = disease_probabilities.get(highest_disease, 0.0)
    if highest_probability >= 0.2:
        questions.append(f"What does my {pretty_disease_name(highest_disease).lower()} risk result mean for my next check-up?")

    useful_conditions = select_clinically_useful_conditions(patient_conditions)
    if not useful_conditions.empty:
        questions.append("Do any of my past conditions change how you would interpret these current results?")

    if not questions:
        questions.extend(
            [
                "Which result should I keep watching over time?",
                "Do any of these values need repeat testing before my next routine visit?",
                "Is there anything I should start doing now to lower long-term risk?",
            ]
        )

    return questions[:5]


def build_symptom_helper(symptom_text: str) -> str:
    normalized_text = symptom_text.strip().lower()
    if not normalized_text:
        return "Type symptoms in your own words, like 'headache and dizziness for 2 days' or 'tired, thirsty, and urinating often'."

    urgent_keywords = ["chest pain", "trouble breathing", "shortness of breath", "fainting", "passed out", "severe weakness"]
    symptom_to_question = []

    if "headache" in normalized_text:
        symptom_to_question.append("- Could these headaches be related to blood pressure, stress, dehydration, or something else?")
    if "dizzy" in normalized_text or "dizziness" in normalized_text:
        symptom_to_question.append("- Could my dizziness be related to blood pressure, hydration, blood sugar, or another issue?")
    if "thirst" in normalized_text or "thirsty" in normalized_text:
        symptom_to_question.append("- Could increased thirst be related to blood sugar or dehydration?")
    if "urinating" in normalized_text or "pee" in normalized_text or "frequent urination" in normalized_text:
        symptom_to_question.append("- Could frequent urination be related to blood sugar, infection, or fluid balance?")
    if "tired" in normalized_text or "fatigue" in normalized_text:
        symptom_to_question.append("- Could my fatigue be related to sleep, stress, blood sugar, anemia, or another cause?")
    if "swelling" in normalized_text:
        symptom_to_question.append("- Could this swelling be related to blood pressure, kidney function, circulation, or something else?")

    if not symptom_to_question:
        symptom_to_question.append("- What are the most likely causes of these symptoms, and what should be checked first?")
        symptom_to_question.append("- Do these symptoms suggest I should come in soon or monitor them at home first?")

    urgent_note = ""
    if any(keyword in normalized_text for keyword in urgent_keywords):
        urgent_note = (
            "Seek urgent medical attention if these symptoms are severe, sudden, or getting worse, especially chest pain, trouble breathing, or fainting."
        )

    return "\n".join(
        [
            "How to describe it to a doctor",
            f"- I am having: {symptom_text.strip()}",
            "- When did it start, how often does it happen, and is it getting worse?",
            "- Are there any triggers, like exercise, food, stress, or not sleeping well?",
            "",
            "Questions you can ask",
            *symptom_to_question,
            "",
            "Safety note",
            urgent_note or "If symptoms are severe, sudden, or clearly getting worse, seek medical care sooner.",
        ]
    )


def build_risk_help(disease_probabilities: dict, latest_values: dict) -> str:
    highest_disease = max(disease_probabilities, key=disease_probabilities.get)
    highest_probability = disease_probabilities.get(highest_disease, 0.0)
    highest_label = pretty_disease_name(highest_disease)
    reasons = build_clinical_flags(latest_values, disease_probabilities)
    personal_focus = build_personal_focus(latest_values, disease_probabilities)

    return "\n".join(
        [
            "What this means for you",
            f"- The main area the app is watching right now is {highest_label.lower()} at {highest_probability * 100:.1f}%.",
            "- This is not a diagnosis. It is a signal that this part of your record deserves more attention.",
            "",
            "What in your record is driving that score",
            *[f"- {item}" for item in personal_focus],
            "",
            "Other supporting details",
            *[f"- {item}" for item in reasons[:4]],
            "",
            "Best next step for you",
            "- Compare this result with future readings and bring it to your next doctor visit.",
        ]
    )


def build_next_step_help(disease_probabilities: dict, latest_values: dict) -> str:
    steps = build_next_steps(max(disease_probabilities.values()) if disease_probabilities else 0.0, latest_values)
    personal_focus = build_personal_focus(latest_values, disease_probabilities)
    return "\n".join(
        [
            "What you can do next for your own record",
            "- Start with the item that best matches the values shown for you right now.",
            "",
            "Why the app is saying this",
            *[f"- {item}" for item in personal_focus],
            "",
            "Suggested next steps",
            *[f"- {step}" for step in steps[:5]],
            "",
            "When to contact a doctor sooner",
            "- If symptoms are new, worsening, severe, or worrying, seek medical advice sooner rather than waiting.",
        ]
    )


def build_trend_help(latest_values: dict) -> str:
    summary_lines = build_record_summary(latest_values)
    return "\n".join(
        [
            "How to read your trend page",
            "- Your latest values are a snapshot of where things stand right now.",
            "- Trend charts are more useful when the same test appears more than once over time.",
            "",
            "What your current record shows",
            *[f"- {line}" for line in summary_lines[:4]],
            "",
            "Good question for your visit",
            "- Has this value changed enough over time to matter for me?",
        ]
    )


def build_report_help() -> str:
    return "\n".join(
        [
            "What this report is for",
            "- It gives a short summary of your recent values, risk checks, and relevant history in one place.",
            "- You can download it and share it with a doctor or online health service.",
            "- It is most useful as a conversation starter, not as a final diagnosis.",
            "",
            "What to share with a doctor",
            "- The downloaded report",
            "- Any symptoms you are having",
            "- Questions about values that are above the ideal range",
        ]
    )


def build_trust_and_limits() -> list[str]:
    return [
        "These results are early risk signals from the current health record, not a final diagnosis.",
        "This app is based on synthetic patient data and the measurements available here.",
        "One value alone may not tell the whole story, so trends and follow-up testing matter.",
        "A doctor should interpret these results together with symptoms, history, and repeat labs when needed.",
    ]


def build_selected_patient_context(
    patient_row: pd.Series,
    features_df: pd.DataFrame,
    models: dict,
    latest_values_lookup: dict,
    condition_lookup: dict,
    encounter_lookup: dict,
    empty_conditions: pd.DataFrame,
    empty_encounters: pd.DataFrame,
    uploaded_reports: list[dict] | None = None,
):
    patient_id = patient_row["Id"]
    patient_conditions = condition_lookup.get(str(patient_id), empty_conditions.copy())
    patient_encounters = encounter_lookup.get(str(patient_id), empty_encounters.copy())
    patient_features = features_df.loc[features_df["Id"] == patient_id].drop(columns=["Id"])
    patient_features = apply_uploaded_values_to_features(patient_features, uploaded_reports)
    model_probabilities = get_disease_probabilities(models, patient_features)
    latest_values = latest_values_lookup.get(str(patient_id), {}).copy()
    latest_values.update(combine_uploaded_report_values(uploaded_reports))
    disease_probabilities = build_record_based_disease_scores(
        latest_values,
        patient_conditions,
        model_probabilities,
        patient_row,
        patient_features,
    )
    highest_probability = max(disease_probabilities.values()) if disease_probabilities else 0.0
    systolic_value = latest_values.get("Systolic blood pressure")
    missing_measure_map = {
        "Blood sugar": "glucose_missing",
        "BMI": "bmi_missing",
        "Creatinine": "creatinine_missing",
        "Systolic blood pressure": "systolic_bp_missing",
    }
    missing_measure_labels = []
    if not patient_features.empty:
        feature_row = patient_features.iloc[0]
        for label, column_name in missing_measure_map.items():
            if float(feature_row.get(column_name, 0.0)) >= 0.5 and label not in latest_values:
                missing_measure_labels.append(label)

    return {
        "disease_probabilities": disease_probabilities,
        "diabetes_probability": disease_probabilities.get("diabetes", 0.0),
        "highest_probability": highest_probability,
        "latest_values": latest_values,
        "summary_lines": build_record_summary(latest_values),
        "next_steps": build_next_steps(highest_probability, latest_values),
        "risk_reasons": build_risk_reasons(patient_row, latest_values, highest_probability),
        "patient_conditions": patient_conditions,
        "patient_encounters": patient_encounters,
        "overall_score": calculate_overall_health_score(disease_probabilities),
        "bp_value_text": f"{systolic_value:.0f} mmHg" if systolic_value is not None else "Not available",
        "first_name": patient_row["FIRST"] if pd.notna(patient_row["FIRST"]) else "Patient",
        "uploaded_report_active": bool(combine_uploaded_report_values(uploaded_reports)),
        "missing_measure_labels": missing_measure_labels,
    }


def choose_demo_patient(patients_df: pd.DataFrame) -> pd.Series:
    demo_ready_patients = patients_df.loc[patients_df.get("measure_coverage", 0) >= 4]
    if demo_ready_patients.empty:
        demo_ready_patients = patients_df

    preferred_match = demo_ready_patients.loc[demo_ready_patients["Id"].astype(str) == DEFAULT_DEMO_PATIENT_ID]
    if not preferred_match.empty:
        return preferred_match.iloc[0]
    return demo_ready_patients.iloc[0]


def generate_patient_help_response(
    user_text: str,
    latest_values: dict,
    disease_probabilities: dict,
    patient_conditions: pd.DataFrame,
) -> str:
    normalized_text = user_text.strip().lower()
    if not normalized_text:
        return "Type a question like 'What does creatinine mean?', 'What should I ask my doctor?', or 'I have headache and dizziness'."

    if "ask my doctor" in normalized_text or "questions for my doctor" in normalized_text:
        questions = build_questions_for_doctor(latest_values, disease_probabilities, patient_conditions)
        return "\n".join(
            [
                "Questions based on your current record",
                *[f"- {question}" for question in questions],
            ]
        )

    if "what does my risk mean" in normalized_text or "why is my risk" in normalized_text or "risk mean" in normalized_text:
        return build_risk_help(disease_probabilities, latest_values)

    if "what should i do next" in normalized_text or "what do i do next" in normalized_text or "next step" in normalized_text:
        return build_next_step_help(disease_probabilities, latest_values)

    if "trend" in normalized_text or "changed over time" in normalized_text or "over time" in normalized_text:
        return build_trend_help(latest_values)

    if "my health" in normalized_text or "my record" in normalized_text or "my data" in normalized_text or "summary" in normalized_text:
        personal_focus = build_personal_focus(latest_values, disease_probabilities)
        return "\n".join(
            [
                "What your record is showing right now",
                *[f"- {item}" for item in personal_focus],
                "",
                "What to do with this information",
                "- Use it to decide what to ask at your next check-up and what values to track over time.",
            ]
        )

    if "report" in normalized_text or "download" in normalized_text or "share with doctor" in normalized_text:
        return build_report_help()

    known_term_match = None
    for known_term in TERM_EXPLANATIONS:
        if known_term in normalized_text:
            known_term_match = known_term
            break

    symptom_keywords = [
        "pain", "headache", "dizziness", "dizzy", "tired", "fatigue", "swelling",
        "thirst", "thirsty", "urinating", "pee", "breathing", "cough", "fever",
        "nausea", "vomit", "weakness", "chest", "symptom",
    ]
    symptom_like = any(keyword in normalized_text for keyword in symptom_keywords)

    if known_term_match and ("what is" in normalized_text or "what does" in normalized_text or "mean" in normalized_text or len(normalized_text.split()) <= 4):
        return explain_medical_term(known_term_match)

    if symptom_like:
        return build_symptom_helper(user_text)

    if known_term_match:
        return explain_medical_term(known_term_match)

    return "\n".join(
        [
            "I can help you understand your own record here",
            "- Explain what one of your values means.",
            "- Tell you what in your record is pushing a score up.",
            "- Suggest useful questions for your doctor based on your results.",
            "- Help you understand what to do next.",
            "- Explain how to read your trend page or report.",
            "",
            "Try asking",
            "- What does my blood pressure mean?",
            "- What in my record looks most important right now?",
            "- Why is my risk score like this?",
            "- What should I ask my doctor next?",
        ]
    )


@st.cache_resource
def prepare_app_state():
    patients_df, observations_df, conditions_df, encounters_df = load_data()
    features_df = build_features(patients_df, observations_df)
    labels_df = build_labels(patients_df, conditions_df)
    models = train_model(features_df, labels_df)
    latest_values_lookup = build_latest_values_lookup(observations_df)
    condition_lookup, encounter_lookup = build_patient_history_lookups(conditions_df, encounters_df)

    patients_df = patients_df.copy()
    patients_df["FULL_NAME"] = (patients_df["FIRST"].fillna("") + " " + patients_df["LAST"].fillna("")).str.strip()
    patients_df["DEMO_LABEL"] = patients_df.apply(
        lambda row: f"{row['FULL_NAME']} ({str(row['Id'])[-6:]})",
        axis=1,
    )
    patients_df = patients_df.sort_values("FULL_NAME")

    return (
        patients_df,
        observations_df,
        conditions_df,
        encounters_df,
        features_df,
        models,
        latest_values_lookup,
        condition_lookup,
        encounter_lookup,
    )


(
    patients_df,
    observations_df,
    conditions_df,
    encounters_df,
    features_df,
    models,
    latest_values_lookup,
    condition_lookup,
    encounter_lookup,
) = prepare_app_state()

st.sidebar.markdown("## LiveWell+")
theme_choice = st.sidebar.radio("Background", ["Dark", "Light"], index=0)
apply_theme(theme_choice)

default_patient_row = choose_demo_patient(patients_df)
patient_id = default_patient_row["Id"]
demo_picker_df = patients_df.loc[patients_df.get("measure_coverage", 0) >= 4].copy()
if demo_picker_df.empty:
    demo_picker_df = patients_df.copy()

with st.sidebar.expander("Demo settings"):
    allow_patient_switch = st.checkbox("Change demo patient", value=False, key="allow_demo_patient_switch")
    if allow_patient_switch:
        selected_demo_id = st.selectbox(
            "Demo patient",
            demo_picker_df["Id"].astype(str).tolist(),
            index=int(demo_picker_df.index.get_loc(default_patient_row.name)),
            format_func=lambda selected_id: patients_df.loc[
                patients_df["Id"].astype(str) == str(selected_id), "DEMO_LABEL"
            ].iloc[0],
            key="demo_patient_id",
        )
        patient_row = patients_df.loc[patients_df["Id"].astype(str) == str(selected_demo_id)].iloc[0]
        patient_id = patient_row["Id"]
    else:
        patient_row = default_patient_row

patient_name = patient_row["FULL_NAME"]

st.sidebar.markdown(
    f"""
    <div class="info-card" style="margin-top:0.75rem;padding:0.9rem 1rem;">
        <div class="small-note" style="margin-bottom:0.25rem;">Your profile</div>
        <div style="font-weight:800;font-size:1.05rem;">{patient_name}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_files = st.sidebar.file_uploader("Upload lab report or patient file", type=None, accept_multiple_files=True)
context_cache_key = f"patient_context_{patient_id}"
uploaded_reports_key = f"uploaded_reports_{patient_id}"

if uploaded_files:
    parsed_reports: list[dict] = []
    for uploaded_file in uploaded_files:
        upload_status, upload_message, parsed_upload = validate_uploaded_file(uploaded_file, patient_id)
        getattr(st.sidebar, upload_status)(upload_message)
        if parsed_upload:
            parsed_reports.append(parsed_upload)

    st.session_state[uploaded_reports_key] = parsed_reports
    st.session_state.pop(context_cache_key, None)
elif uploaded_reports_key in st.session_state:
    st.session_state.pop(uploaded_reports_key, None)
    st.session_state.pop(context_cache_key, None)

page = st.sidebar.radio("Navigation", ["Home", "My History", "Health Check", "My Reports"])

empty_conditions = conditions_df.iloc[0:0].copy()
empty_encounters = encounters_df.iloc[0:0].copy()
uploaded_reports = st.session_state.get(uploaded_reports_key, [])

if context_cache_key not in st.session_state:
    st.session_state[context_cache_key] = build_selected_patient_context(
        patient_row,
        features_df,
        models,
        latest_values_lookup,
        condition_lookup,
        encounter_lookup,
        empty_conditions,
        empty_encounters,
        uploaded_reports,
    )

patient_context = st.session_state[context_cache_key]
disease_probabilities = patient_context["disease_probabilities"]
highest_probability = patient_context["highest_probability"]
latest_values = patient_context["latest_values"]
summary_lines = patient_context["summary_lines"]
next_steps = patient_context["next_steps"]
risk_reasons = patient_context["risk_reasons"]
patient_conditions = patient_context["patient_conditions"]
patient_encounters = patient_context["patient_encounters"]
overall_score = patient_context["overall_score"]
bp_value_text = patient_context["bp_value_text"]
first_name = patient_context["first_name"]
uploaded_report_active = patient_context["uploaded_report_active"]
missing_measure_labels = patient_context["missing_measure_labels"]


def render_home():
    overall_tone = tone_from_score(overall_score)
    blood_pressure_tone = tone_from_bp(latest_values)
    highest_risk_tone = tone_from_risk(highest_probability)

    st.markdown(
        f"""
        <div class="hero-card">
            <div style="font-size:2.3rem;font-weight:800;">Welcome, {first_name}!</div>
            <div class="small-note">A simple app to help you understand your health information.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if uploaded_report_active and uploaded_reports:
        st.markdown(
            f"""
            <div class="summary-box">
                Uploaded values from {len(uploaded_reports)} file(s) are now being used in this dashboard view.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if missing_measure_labels:
        missing_text = ", ".join(missing_measure_labels)
        st.info(f"Some scores are based on partial data because this patient does not have recent values for: {missing_text}.")

    intro_col, guide_col = st.columns([1.15, 0.85])
    with intro_col:
        st.markdown(
            """
            <div class="info-card">
                <div class="info-title">How To Use This App</div>
                <ul class="info-list">
                    <li>Start with the summary cards to see the main health signals.</li>
                    <li>Open Health Check to read a plain-language explanation and trend view.</li>
                    <li>Use Smart Patient Help when you want terms explained or questions to ask a doctor.</li>
                    <li>Download My Report if you want one file to share during a visit.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with guide_col:
        trust_items = "".join(f"<li>{item}</li>" for item in build_trust_and_limits())
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">Trust And Limits</div>
                <ul class="info-list">{trust_items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card tone-{overall_tone}">
                <div class="metric-label">Overall Health Score</div>
                <div class="metric-value tone-{overall_tone}">{overall_score} / 100</div>
                <div class="small-note">Calculated from the current patient record</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card tone-{blood_pressure_tone}">
                <div class="metric-label">Latest Blood Pressure</div>
                <div class="metric-value tone-{blood_pressure_tone}">{bp_value_text}</div>
                <div class="small-note">Latest recorded systolic value</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card tone-{highest_risk_tone}">
                <div class="metric-label">Highest Risk Score</div>
                <div class="metric-value tone-{highest_risk_tone}">{highest_probability * 100:.1f}%</div>
                <div class="small-note">Highest score across the health checks below</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Risk Checks")
    disease_columns = st.columns(4)
    ordered_diseases = [
        "kidney_disease",
        "diabetes",
        "cardiovascular_disease",
        "hypertension",
    ]
    for column, disease_name in zip(disease_columns, ordered_diseases):
        disease_probability = disease_probabilities.get(disease_name, 0.0)
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{pretty_disease_name(disease_name)}</div>
                    <div class="metric-value">{disease_probability * 100:.1f}%</div>
                    <div class="small-note">Record-based score from current values and history</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    summary_banner = ""
    if highest_probability >= 0.6:
        summary_banner = '<div class="summary-box">Current record review: at least one health check is in a higher-risk range, so a follow-up review would be appropriate.</div>'
    elif highest_probability >= 0.35:
        summary_banner = '<div class="summary-box">Current record review: there are moderate signals in this patient record, so trend monitoring and follow-up may help.</div>'
    else:
        summary_banner = '<div class="summary-box">Current record review: the latest values in this patient record look more stable overall.</div>'

    summary_items = "".join(f"<li>{line}</li>" for line in summary_lines)
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">Today's Summary</div>
            {summary_banner}
            <ul class="info-list">{summary_items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.05, 0.95])
    with left_col:
        next_step_items = "".join(f"<li>{step}</li>" for step in next_steps)
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">What To Do Next</div>
                <ul class="info-list">{next_step_items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        reason_items = "".join(f"<li>{reason}</li>" for reason in risk_reasons[:4])
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">What We Are Watching</div>
                <ul class="info-list">{reason_items}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_history():
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.subheader("My History")
    st.write("This page shows the most useful recent health history and care visits from your record.")

    useful_conditions = select_clinically_useful_conditions(patient_conditions)

    if not useful_conditions.empty:
        history_conditions = useful_conditions[["START", "DESCRIPTION"]].copy().sort_values("START", ascending=False).head(12)
        history_conditions.columns = ["Date", "Condition"]
        st.write("**Recent health conditions**")
        render_html_table(history_conditions)
    else:
        st.info("No clinically useful recent health conditions were found.")

    if not patient_encounters.empty:
        history_encounters = patient_encounters[["START", "DESCRIPTION"]].copy().sort_values("START", ascending=False).head(12)
        history_encounters.columns = ["Date", "Visit"]
        st.write("**Recent doctor visits and care events**")
        render_html_table(history_encounters)
    else:
        st.info("No recent care visits were found.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_health_check():
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.subheader("Health Check")
    st.caption("This page uses the information already in your health record.")

    st.write("**What your recent record shows:**")
    for line in summary_lines:
        st.write(f"- {line}")

    st.write("**Why this may need attention:**")
    for reason in risk_reasons:
        st.write(f"- {reason}")

    st.write("**Simple summary:**")
    bp_value = latest_values.get("Systolic blood pressure", 0)
    bmi_value = latest_values.get("BMI", 0)
    if (
        latest_values.get("Blood sugar", 0) >= 126
        or latest_values.get("Creatinine", 0) > 1.3
        or bp_value >= 140
        or bmi_value >= 30
        or highest_probability >= 0.6
    ):
        st.markdown(
            '<div class="summary-box summary-warning">This record shows stronger warning signals in the latest values. It does not confirm a disease, but follow-up would be reasonable.</div>',
            unsafe_allow_html=True,
        )
    elif (
        latest_values.get("Blood sugar", 0) >= 100
        or latest_values.get("Creatinine", 0) > 1.1
        or bp_value >= 120
        or bmi_value >= 25
        or highest_probability >= 0.35
    ):
        st.markdown(
            '<div class="summary-box summary-warning">This record shows a few mild or moderate signals that are worth watching over time.</div>',
            unsafe_allow_html=True,
        )
    elif latest_values:
        st.markdown(
            '<div class="summary-box summary-success">The latest record looks relatively stable overall, based on the values available here.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="summary-box summary-warning">There are not enough recent record values yet to build a stronger health check.</div>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.subheader("Health Trend")
    trend_options = {
        "Blood sugar": ["glucose"],
        "BMI": ["body mass index", "bmi"],
        "Creatinine": ["creatinine"],
        "Systolic blood pressure": ["systolic blood pressure"],
    }
    combined_uploaded_values = combine_uploaded_report_values(uploaded_reports)
    trend_df = observations_df.copy()
    trend_df = trend_df[trend_df["PATIENT"] == patient_id]
    trend_df["VALUE"] = pd.to_numeric(trend_df["VALUE"], errors="coerce")
    date_column = "DATE" if "DATE" in trend_df.columns else "START"
    trend_df[date_column] = pd.to_datetime(trend_df[date_column], errors="coerce")
    trend_df = trend_df.dropna(subset=["VALUE", date_column])

    available_trends = []
    for label, patterns in trend_options.items():
        has_points = trend_df["DESCRIPTION"].fillna("").str.lower().apply(
            lambda text: any(pattern in text for pattern in patterns)
        ).any()
        if has_points or label in combined_uploaded_values:
            available_trends.append(label)

    trend_labels = available_trends if available_trends else list(trend_options.keys())
    selected_trend = st.selectbox("Choose a measure", trend_labels)
    trend_patterns = trend_options[selected_trend]

    trend_df = trend_df[
        trend_df["DESCRIPTION"].fillna("").str.lower().apply(
            lambda text: any(pattern in text for pattern in trend_patterns)
        )
    ].sort_values(date_column)

    if selected_trend in combined_uploaded_values:
        latest_uploaded_at = None
        for report in uploaded_reports:
            if selected_trend in report.get("values", {}):
                report_uploaded_at = pd.to_datetime(report.get("uploaded_at"), errors="coerce")
                if latest_uploaded_at is None or (pd.notna(report_uploaded_at) and report_uploaded_at > latest_uploaded_at):
                    latest_uploaded_at = report_uploaded_at
        if latest_uploaded_at is None or pd.isna(latest_uploaded_at):
            latest_uploaded_at = pd.Timestamp.now()

        uploaded_row = pd.DataFrame(
            [
                {
                    date_column: latest_uploaded_at,
                    "VALUE": float(combined_uploaded_values[selected_trend]),
                    "DESCRIPTION": f"Uploaded {selected_trend}",
                }
            ]
        )
        trend_df = pd.concat([trend_df, uploaded_row], ignore_index=True).sort_values(date_column)

    if not trend_df.empty:
        recent_trend = trend_df.tail(10).copy()
        recent_trend["Label"] = recent_trend[date_column].dt.strftime("%Y-%m-%d")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(recent_trend["Label"], recent_trend["VALUE"], color="#2563eb")
        ax.set_title(f"{selected_trend} over time")
        ax.set_ylabel("Value")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
    else:
        st.info(f"No trend chart data was found for {selected_trend.lower()}, even though other latest values may still be available in the record.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.subheader("Smart Patient Help")
    st.caption("Ask about your own results here. This helper uses the values and scores already shown in your record.")

    smart_help_input_key = f"smart_help_input_{patient_id}"
    smart_help_result_key = f"smart_help_result_{patient_id}"

    if smart_help_input_key not in st.session_state:
        st.session_state[smart_help_input_key] = ""

    suggestion_col1, suggestion_col2, suggestion_col3 = st.columns(3)
    with suggestion_col1:
        if st.button("What matters most right now?", key=f"smart_suggestion_term_{patient_id}"):
            st.session_state[smart_help_input_key] = "What in my record looks most important right now?"
    with suggestion_col2:
        if st.button("What should I ask my doctor?", key=f"smart_suggestion_doctor_{patient_id}"):
            st.session_state[smart_help_input_key] = "What should I ask my doctor about my results?"
    with suggestion_col3:
        if st.button("Explain my blood pressure", key=f"smart_suggestion_symptom_{patient_id}"):
            st.session_state[smart_help_input_key] = "What does my blood pressure mean?"

    with st.form(key=f"smart_help_form_{patient_id}", clear_on_submit=False):
        user_text = st.text_area(
            "Ask Smart Patient Help",
            key=smart_help_input_key,
            height=120,
            placeholder="Type your question here...",
        )
        asked = st.form_submit_button("Ask")

    if asked:
        st.session_state[smart_help_result_key] = generate_patient_help_response(
            user_text,
            latest_values,
            disease_probabilities,
            patient_conditions,
        )

    if st.session_state.get(smart_help_result_key):
        st.markdown(report_preview_html(st.session_state[smart_help_result_key]), unsafe_allow_html=True)
    else:
        st.info("Try a question like: What in my record looks most important right now? What should I ask my doctor about my results? What does my blood pressure mean?")

    if uploaded_reports:
        st.markdown("**Uploaded Report Summary**")
        st.markdown(report_preview_html(build_uploaded_reports_summary(uploaded_reports)), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_reports():
    st.markdown('<div class="section-shell">', unsafe_allow_html=True)
    st.subheader("My Reports")
    st.write("Download a practical summary you can share with a doctor, upload to an online health service, or paste into an AI tool for follow-up questions.")

    report_text = build_report_text(
        patient_name,
        patient_row,
        latest_values,
        disease_probabilities,
        risk_reasons,
        next_steps,
        patient_conditions,
        patient_encounters,
    )
    st.download_button(
        label="Download My Report",
        data=download_bytes(report_text),
        file_name=f"{patient_name.replace(' ', '_').lower()}_report.txt",
        mime="text/plain",
    )

    st.write("**Preview**")
    st.markdown(report_preview_html(report_text), unsafe_allow_html=True)

    if uploaded_reports:
        st.write("**Uploaded Hospital-Style Files**")
        st.markdown(report_preview_html(build_uploaded_reports_summary(uploaded_reports)), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


if page == "Home":
    render_home()
elif page == "My History":
    render_history()
elif page == "Health Check":
    render_health_check()
else:
    render_reports()
