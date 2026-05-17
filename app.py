import streamlit as st
import pandas as pd
import numpy as np
import warnings
import os
import joblib

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📡",
    layout="wide",
)

st.markdown("""
<style>
/* Remove default top padding */
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }

/* Title row */
.title-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 2px;
}
.title-row h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #1a56db, #7e3af2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.subtitle {
    color: #888;
    font-size: 0.88rem;
    margin: 0 0 14px 0;
}

/* Section headers */
.sec {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.3px;
    text-transform: uppercase;
    color: #7e3af2;
    border-bottom: 1px solid #2d2d2d;
    padding-bottom: 3px;
    margin: 14px 0 8px;
}

/* Result cards */
.card-high {
    background: linear-gradient(135deg,#2d0000,#1a0000);
    border-left: 5px solid #e3342f;
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 14px;
}
.card-low {
    background: linear-gradient(135deg,#002d14,#001a0d);
    border-left: 5px solid #38a169;
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 14px;
}
.card-high h2 { color:#fc8181; font-size:1.5rem; margin:0 0 4px; }
.card-low  h2 { color:#68d391; font-size:1.5rem; margin:0 0 4px; }
.card-high p, .card-low p { color:#ccc; font-size:0.95rem; margin:0; }
.pct-high { font-size:2.8rem; font-weight:900; color:#fc8181; line-height:1.1; }
.pct-low  { font-size:2.8rem; font-weight:900; color:#68d391; line-height:1.1; }
</style>
""", unsafe_allow_html=True)

# ── Compact title ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-row"><h1>📡 Telco Churn Predictor</h1></div>
<p class="subtitle">AI-powered customer retention · Artificial Neural Network</p>
""", unsafe_allow_html=True)
st.divider()

# ── Model loading ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = os.path.join(BASE, "churn_ann_model.h5")
SCALER_PATH = os.path.join(BASE, "scaler.pkl")
OHE_PATH    = os.path.join(BASE, "ohe_encoder.pkl")
CSV_CANDIDATES = [
    os.path.join(BASE, "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    os.path.join(BASE, "telco_churn.csv"),
    os.path.join(BASE, "churn.csv"),
]

def _preprocess(df, ohe, fit_ohe=True):
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"]        = df["Churn"].map({"Yes": 1, "No": 0})
    df.dropna(inplace=True); df.drop_duplicates(inplace=True)
    df.replace("No internet service", "No", inplace=True)
    df.replace("No phone service",    "No", inplace=True)
    for col in ["Partner","Dependents","PhoneService","MultipleLines",
                "OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
                "StreamingTV","StreamingMovies","PaperlessBilling"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes":1,"No":0})
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male":1,"Female":0})
    obj_cols = ["InternetService","Contract","PaymentMethod"]
    enc = ohe.fit_transform(df[obj_cols]) if fit_ohe else ohe.transform(df[obj_cols])
    df  = pd.concat([df.drop(obj_cols,axis=1),
                     pd.DataFrame(enc,columns=ohe.get_feature_names_out(obj_cols),
                                  index=df.index)],axis=1)
    return df

@st.cache_resource(show_spinner="⚡ Loading model…")
def load_artifacts():
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    # Load saved
    if all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, OHE_PATH]):
        try:
            model  = load_model(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            ohe    = joblib.load(OHE_PATH)
            base   = ["gender","SeniorCitizen","Partner","Dependents","tenure",
                      "PhoneService","MultipleLines","OnlineSecurity","OnlineBackup",
                      "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
                      "PaperlessBilling","MonthlyCharges","TotalCharges"]
            feat   = base + ohe.get_feature_names_out(
                        ["InternetService","Contract","PaymentMethod"]).tolist()
            return model, scaler, ohe, feat
        except Exception:
            pass

    # Train fresh
    csv = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    if csv is None:
        return None
    ohe = OneHotEncoder(drop="first", sparse_output=False)
    df  = _preprocess(pd.read_csv(csv), ohe, fit_ohe=True)
    X, y  = df.drop("Churn",axis=1), df["Churn"]
    feat  = X.columns.tolist()
    X_tr,_,y_tr,_ = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_tr)
    X_sm,y_sm = SMOTE(random_state=42).fit_resample(X_sc, y_tr)
    model = Sequential([
        Dense(64,activation="relu",input_dim=X_sm.shape[1]),
        BatchNormalization(),Dropout(0.3),
        Dense(32,activation="relu"),BatchNormalization(),Dropout(0.3),
        Dense(16,activation="relu"),Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    model.fit(X_sm,y_sm,epochs=100,batch_size=32,validation_split=0.2,
              callbacks=[EarlyStopping(monitor="val_loss",patience=8,
                                       restore_best_weights=True,verbose=0)],verbose=0)
    model.save(MODEL_PATH); joblib.dump(scaler,SCALER_PATH); joblib.dump(ohe,OHE_PATH)
    return model, scaler, ohe, feat

result = load_artifacts()
if result is None:
    st.error("CSV not found. Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` next to app.py")
    st.stop()

model, scaler, ohe, feature_names = result
yn = {"No":0,"Yes":1}

# ── Input form — inline below title ───────────────────────────────────────────
with st.form("churn_form"):

    st.markdown('<div class="sec">👤 Customer Info</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    gender     = c1.selectbox("Gender",         ["Male","Female"])
    senior     = c2.selectbox("Senior Citizen", ["No","Yes"])
    partner    = c3.selectbox("Partner",        ["No","Yes"])
    dependents = c4.selectbox("Dependents",     ["No","Yes"])
    tenure     = c5.slider("Tenure (months)",   0, 72, 12)

    st.markdown('<div class="sec">📞 Services</div>', unsafe_allow_html=True)
    s1,s2,s3,s4,s5 = st.columns(5)
    phone_service    = s1.selectbox("Phone Service",    ["Yes","No"])
    multiple_lines   = s2.selectbox("Multiple Lines",   ["No","Yes"])
    internet_service = s3.selectbox("Internet Service", ["DSL","Fiber optic","No"])
    online_security  = s4.selectbox("Online Security",  ["No","Yes"])
    online_backup    = s5.selectbox("Online Backup",    ["No","Yes"])

    s6,s7,s8,s9,_ = st.columns(5)
    device_prot      = s6.selectbox("Device Protection",["No","Yes"])
    tech_support     = s7.selectbox("Tech Support",     ["No","Yes"])
    streaming_tv     = s8.selectbox("Streaming TV",     ["No","Yes"])
    streaming_movies = s9.selectbox("Streaming Movies", ["No","Yes"])

    st.markdown('<div class="sec">💳 Billing</div>', unsafe_allow_html=True)
    b1,b2,b3,b4,b5 = st.columns(5)
    contract        = b1.selectbox("Contract",       ["Month-to-month","One year","Two year"])
    payment         = b2.selectbox("Payment Method", ["Electronic check","Mailed check",
                                                      "Bank transfer (automatic)",
                                                      "Credit card (automatic)"])
    paperless       = b3.selectbox("Paperless Billing",["Yes","No"])
    monthly_charges = b4.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total_charges   = b5.number_input("Total Charges ($)",   0.0, 10000.0,
                                       float(65.0*12), step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔮  Predict Churn", use_container_width=True,
                                      type="primary")

# ── Prediction result ──────────────────────────────────────────────────────────
if submitted:
    row = {
        "gender":           1 if gender=="Male" else 0,
        "SeniorCitizen":    yn[senior],
        "Partner":          yn[partner],
        "Dependents":       yn[dependents],
        "tenure":           tenure,
        "PhoneService":     yn[phone_service],
        "MultipleLines":    yn[multiple_lines],
        "OnlineSecurity":   yn[online_security],
        "OnlineBackup":     yn[online_backup],
        "DeviceProtection": yn[device_prot],
        "TechSupport":      yn[tech_support],
        "StreamingTV":      yn[streaming_tv],
        "StreamingMovies":  yn[streaming_movies],
        "PaperlessBilling": yn[paperless],
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }
    enc_vals = ohe.transform(pd.DataFrame(
        [[internet_service, contract, payment]],
        columns=["InternetService","Contract","PaymentMethod"]
    ))
    for col,val in zip(ohe.get_feature_names_out(
            ["InternetService","Contract","PaymentMethod"]), enc_vals[0]):
        row[col] = val

    inp = pd.DataFrame([row])
    for c in feature_names:
        if c not in inp.columns: inp[c] = 0
    inp = inp[feature_names]

    prob     = float(model.predict(scaler.transform(inp), verbose=0)[0][0])
    prob_pct = prob * 100

    # Result + snapshot side by side
    left, right = st.columns([3, 2])

    with left:
        if prob > 0.5:
            st.markdown(f"""
            <div class="card-high">
              <h2>⚠️ HIGH CHURN RISK</h2>
              <p>This customer is <strong style="color:#fc8181">likely to leave.</strong>
                 Take immediate retention action.</p>
              <div class="pct-high">{prob_pct:.1f}%</div>
              <p style="font-size:.8rem;color:#999;margin-top:2px">Churn Probability</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-low">
              <h2>✅ LOW CHURN RISK</h2>
              <p>This customer is <strong style="color:#68d391">likely to stay.</strong>
                 Keep up regular engagement.</p>
              <div class="pct-low">{prob_pct:.1f}%</div>
              <p style="font-size:.8rem;color:#999;margin-top:2px">Churn Probability</p>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>**Probability Meter**", unsafe_allow_html=True)
        st.progress(int(prob_pct))

    with right:
        st.markdown("**Customer Snapshot**")
        st.metric("Tenure",          f"{tenure} months")
        st.metric("Monthly Charges", f"${monthly_charges:.2f}")
        st.metric("Total Charges",   f"${total_charges:.2f}")
        st.metric("Contract",        contract)
        st.metric("Internet",        internet_service)

    # Retention advice
    st.markdown("---")
    if prob > 0.75:
        st.error("🚨 **Urgent:** Offer a significant discount or free upgrade immediately.")
    elif prob > 0.5:
        st.warning("💡 **Tip:** Proactive outreach — a contract upgrade or support call may retain this customer.")
    elif prob > 0.25:
        st.info("💡 **Tip:** Mild risk detected. Consider a loyalty reward or check-in.")
    else:
        st.success("✅ Customer is stable. Standard engagement is sufficient.")