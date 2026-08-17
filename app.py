"""
Sepsis Risk Prediction — Flask deployment app
------------------------------------------------
Loads the trained model bundle (sepsis_model.pkl) exported from the EDA/ML
notebook and serves a dynamic web form where a clinician can enter a
patient's current vitals/labs and get a real-time sepsis risk prediction.

Run:
    pip install flask numpy pandas scikit-learn
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_PATH = 'sepsis_model.pkl'

with open(MODEL_PATH, 'rb') as f:
    BUNDLE = pickle.load(f)

MODEL = BUNDLE['model']
MODEL_NAME = BUNDLE['model_name']
NEEDS_SCALING = BUNDLE['needs_scaling']
NEEDS_IMPUTATION = BUNDLE['needs_imputation']
SCALER = BUNDLE['scaler']
IMPUTER = BUNDLE['imputer']
FEATURE_COLS = BUNDLE['feature_cols']
MEDIAN_VALUES = BUNDLE['median_values']
METRICS = BUNDLE['metrics']

# Field metadata: (label, unit, normal range hint, default value, step)
FIELD_META = {
    'Hour':        ('ICU Hour', 'hr since admission', '', 'ICULOS', 1),
    'HR':          ('Heart Rate', 'bpm', '60–100', 'vital', 1),
    'O2Sat':       ('O2 Saturation', '%', '95–100', 'vital', 1),
    'Temp':        ('Temperature', '°C', '36.1–37.2', 'vital', 0.1),
    'SBP':         ('Systolic BP', 'mmHg', '90–120', 'vital', 1),
    'MAP':         ('Mean Arterial Pressure', 'mmHg', '70–100', 'vital', 1),
    'DBP':         ('Diastolic BP', 'mmHg', '60–80', 'vital', 1),
    'Resp':        ('Respiratory Rate', 'breaths/min', '12–20', 'vital', 1),
    'EtCO2':       ('EtCO2', 'mmHg', '35–45', 'lab', 1),
    'BaseExcess':  ('Base Excess', 'mEq/L', '-2 to 2', 'lab', 0.1),
    'HCO3':        ('Bicarbonate (HCO3)', 'mEq/L', '22–28', 'lab', 0.1),
    'FiO2':        ('FiO2', 'fraction', '0.21–1.0', 'lab', 0.01),
    'pH':          ('Arterial pH', '', '7.35–7.45', 'lab', 0.01),
    'PaCO2':       ('PaCO2', 'mmHg', '35–45', 'lab', 1),
    'SaO2':        ('SaO2', '%', '95–100', 'lab', 1),
    'BUN':         ('Blood Urea Nitrogen', 'mg/dL', '7–20', 'lab', 1),
    'Calcium':     ('Calcium', 'mg/dL', '8.5–10.5', 'lab', 0.1),
    'Chloride':    ('Chloride', 'mEq/L', '96–106', 'lab', 1),
    'Creatinine':  ('Creatinine', 'mg/dL', '0.6–1.3', 'lab', 0.1),
    'Glucose':     ('Glucose', 'mg/dL', '70–140', 'lab', 1),
    'Lactate':     ('Lactate', 'mmol/L', '0.5–2.2', 'lab', 0.1),
    'Magnesium':   ('Magnesium', 'mg/dL', '1.7–2.2', 'lab', 0.1),
    'Potassium':   ('Potassium', 'mEq/L', '3.5–5.0', 'lab', 0.1),
    'AST':              ('AST', 'IU/L', '10–40', 'lab', 1),
    'Alkalinephos':     ('Alkaline Phosphatase', 'IU/L', '44–147', 'lab', 1),
    'Bilirubin_direct': ('Bilirubin (Direct)', 'mg/dL', '0–0.3', 'lab', 0.1),
    'Bilirubin_total':  ('Bilirubin (Total)', 'mg/dL', '0.1–1.2', 'lab', 0.1),
    'TroponinI':        ('Troponin I', 'ng/mL', '0–0.04', 'lab', 0.01),
    'PTT':              ('Partial Thromboplastin Time', 'sec', '25–35', 'lab', 0.1),
    'Fibrinogen':       ('Fibrinogen', 'mg/dL', '200–400', 'lab', 1),
    'Hct':         ('Hematocrit', '%', '38–50', 'lab', 0.1),
    'Hgb':         ('Hemoglobin', 'g/dL', '12–17', 'lab', 0.1),
    'WBC':         ('White Blood Cells', '10^3/µL', '4.5–11', 'lab', 0.1),
    'Platelets':   ('Platelets', '10^3/µL', '150–400', 'lab', 1),
    'Age':         ('Age', 'years', '', 'demo', 1),
    'Gender':      ('Gender', '0 = Female, 1 = Male', '', 'demo', 1),
    'Unit1':       ('MICU Admission', '1 = Yes, 0 = No', '', 'demo', 1),
    'Unit2':       ('SICU Admission', '1 = Yes, 0 = No', '', 'demo', 1),
    'HospAdmTime': ('Hospital Admit Time', 'hrs before ICU admit', '', 'demo', 0.1),
    'ICULOS':      ('ICU Length of Stay', 'hours', '', 'demo', 1),
}

# Fallback for any feature column the model expects but that isn't
# explicitly described above yet (prevents KeyError, defaults to 'lab').
for _col in FEATURE_COLS:
    if _col not in FIELD_META:
        FIELD_META[_col] = (_col, '', '', 'lab', 1)

VITAL_FIELDS = [c for c in FEATURE_COLS if FIELD_META[c][3] == 'vital']
LAB_FIELDS = [c for c in FEATURE_COLS if FIELD_META[c][3] == 'lab']
DEMO_FIELDS = [c for c in FEATURE_COLS if FIELD_META[c][3] in ('demo', 'ICULOS')]


def build_field_list(names):
    return [{
        'name': n,
        'label': FIELD_META[n][0],
        'unit': FIELD_META[n][1],
        'range': FIELD_META[n][2],
        'step': FIELD_META[n][4],
        'default': round(float(MEDIAN_VALUES[n]), 2),
    } for n in names]


def predict_risk(form):
    """Build the feature vector from submitted form data (blanks -> median),
    run it through the same preprocessing used at training time, and return
    a probability + risk tier."""
    row = []
    for col in FEATURE_COLS:
        raw = form.get(col, '').strip()
        if raw == '':
            val = np.nan if not NEEDS_IMPUTATION else MEDIAN_VALUES[col]
            if MODEL_NAME == 'HistGradientBoosting':
                val = np.nan  # HGB handles missing values natively
        else:
            try:
                val = float(raw)
            except ValueError:
                val = MEDIAN_VALUES[col]
        row.append(val)

    X = pd.DataFrame([row], columns=FEATURE_COLS)

    if NEEDS_IMPUTATION:
        X = pd.DataFrame(IMPUTER.transform(X), columns=FEATURE_COLS)
    if NEEDS_SCALING:
        X = SCALER.transform(X.values)

    proba = float(MODEL.predict_proba(X)[0, 1])

    if proba < 0.20:
        tier, tier_class = 'Low Risk', 'risk-low'
    elif proba < 0.50:
        tier, tier_class = 'Moderate Risk', 'risk-moderate'
    elif proba < 0.75:
        tier, tier_class = 'High Risk', 'risk-high'
    else:
        tier, tier_class = 'Critical Risk', 'risk-critical'

    return proba, tier, tier_class


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    submitted_values = {}

    if request.method == 'POST':
        submitted_values = {k: request.form.get(k, '') for k in FEATURE_COLS}
        proba, tier, tier_class = predict_risk(request.form)
        result = {
            'probability': round(proba * 100, 1),
            'tier': tier,
            'tier_class': tier_class,
        }

    return render_template(
        'index.html',
        vital_fields=build_field_list(VITAL_FIELDS),
        lab_fields=build_field_list(LAB_FIELDS),
        demo_fields=build_field_list(DEMO_FIELDS),
        result=result,
        submitted_values=submitted_values,
        model_name=MODEL_NAME,
        metrics=METRICS,
    )


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint for programmatic / AJAX use."""
    data = request.get_json(force=True) or {}
    proba, tier, tier_class = predict_risk(data)
    return {
        'probability': round(proba, 4),
        'risk_percent': round(proba * 100, 1),
        'risk_tier': tier,
        'model': MODEL_NAME,
    }


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
