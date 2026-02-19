import os, sys, warnings, json, tempfile, posixpath
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import joblib
import tarfile
import boto3
import shap

# ----------------------------
# Setup & Path Configuration
# ----------------------------
warnings.simplefilter("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.feature_utils import extract_features

# ----------------------------
# Secrets (Streamlit Cloud)
# ----------------------------
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

REGION = "us-east-1"

# ----------------------------
# Cached AWS session + runtime client
# ----------------------------
@st.cache_resource
def get_boto_session(_id, _secret, _token):
    return boto3.Session(
        aws_access_key_id=_id,
        aws_secret_access_key=_secret,
        aws_session_token=_token,
        region_name=REGION
    )

session = get_boto_session(aws_id, aws_secret, aws_token)

@st.cache_resource
def get_runtime_client(_session):
    return _session.client("sagemaker-runtime")

runtime = get_runtime_client(session)

# ----------------------------
# Data & Model Config
# ----------------------------
df_features = extract_features()

MODEL_INFO = {
    "endpoint": aws_endpoint,
    # IMPORTANT: make sure in S3 this file is actually a .joblib (recommended),
    # not a .shap file using shap.Explainer.save()
    "explainer": "explainer.joblib",
    "keys": ["WMT", "TGT", "DEXJPUS", "DEXCAUS", "SP500", "DJIA", "VIXCLS"],
    "inputs": [{"name": k, "type": "number", "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
               for k in ["WMT", "TGT", "DEXCAUS", "DEXJPUS", "SP500", "DJIA", "VIXCLS"]]
}

# ----------------------------
# SHAP Explainer Loader (S3 -> local temp -> joblib.load)
# ----------------------------
@st.cache_resource
def load_shap_explainer(_session, bucket, s3_key, local_filename):
    s3 = _session.client("s3")
    local_path = os.path.join(tempfile.gettempdir(), local_filename)

    if not os.path.exists(local_path):
        s3.download_file(Bucket=bucket, Key=s3_key, Filename=local_path)

    # safest: joblib
    return joblib.load(local_path)

# ----------------------------
# Call SageMaker endpoint via boto3 runtime
# ----------------------------
def call_model_api(input_df: pd.DataFrame):
    """
    Expects your SageMaker inference.py to accept application/json like:
      {"data": [{"col1":..., "col2":...}, ...]}
    and return:
      {"predictions":[...]}
    """
    try:
        payload = {"data": input_df.to_dict(orient="records")}
        resp = runtime.invoke_endpoint(
            EndpointName=MODEL_INFO["endpoint"],
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload),
        )
        out = json.loads(resp["Body"].read().decode("utf-8"))
        pred_val = float(out["predictions"][-1])  # last row corresponds to appended row
        return round(pred_val, 4), 200
    except Exception as e:
        return f"Error calling endpoint: {str(e)}", 500

# ----------------------------
# Local Explainability
# ----------------------------
def display_explanation(input_df, _session, bucket):
    st.subheader("🔍 Decision Transparency (SHAP)")

    # Update this to match where you stored the explainer in S3:
    # e.g. "explainer/explainer.joblib"
    s3_key = posixpath.join("explainer", MODEL_INFO["explainer"])

    try:
        explainer = load_shap_explainer(_session, bucket, s3_key, MODEL_INFO["explainer"])
        shap_values = explainer(input_df)

        # Waterfall plot for the LAST row (your user input is last row)
        idx = len(input_df) - 1

        plt.figure(figsize=(10, 4))
        shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
        st.pyplot(plt.gcf())
        plt.close()

        # Top feature info (simple + safe)
        top_i = int(np.argmax(np.abs(shap_values[idx].values)))
        top_feature = shap_values[idx].feature_names[top_i]
        st.info(f"**Business Insight:** The most influential factor was **{top_feature}**.")

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    st.subheader("Inputs")
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"].replace("_", " ").upper(),
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"])
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    # Arrange inputs in key order
    data_row = [user_inputs[k] for k in MODEL_INFO["keys"]]

    # Build input df by appending one row to your feature template
    base_df = df_features.copy()
    input_df = pd.concat([base_df, pd.DataFrame([data_row], columns=base_df.columns)], ignore_index=True)

    res, status = call_model_api(input_df)
    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df, session, aws_bucket)
    else:
        st.error(res)
