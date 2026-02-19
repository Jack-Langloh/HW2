import warnings, json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tempfile, os

import boto3
import joblib
import shap

warnings.simplefilter("ignore")

# ----------------------------
# Secrets
# ----------------------------
aws_id = st.secrets["aws_credentials"]["AWS_ACCESS_KEY_ID"]
aws_secret = st.secrets["aws_credentials"]["AWS_SECRET_ACCESS_KEY"]
aws_token = st.secrets["aws_credentials"]["AWS_SESSION_TOKEN"]
aws_bucket = st.secrets["aws_credentials"]["AWS_BUCKET"]
aws_endpoint = st.secrets["aws_credentials"]["AWS_ENDPOINT"]

REGION = "us-east-1"

# ----------------------------
# AWS Session + Runtime Client
# ----------------------------
@st.cache_resource
def get_runtime():
    session = boto3.Session(
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name=REGION
    )
    return session.client("sagemaker-runtime")

runtime = get_runtime()

# ----------------------------
# Load feature template (LOCAL CSV)
# ----------------------------
df_features = pd.read_csv("df_features.csv")

MODEL_INFO = {
    "endpoint": aws_endpoint,
    "explainer": "explainer.joblib",
    "keys": ["WMT", "TGT", "DEXJPUS", "DEXCAUS", "SP500", "DJIA", "VIXCLS"],
    "inputs": [{"name": k, "min": -1.0, "max": 1.0, "default": 0.0, "step": 0.01}
               for k in ["WMT", "TGT", "DEXCAUS", "DEXJPUS", "SP500", "DJIA", "VIXCLS"]]
}

# ----------------------------
# SHAP Explainer Loader
# ----------------------------
@st.cache_resource
def load_shap():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_id,
        aws_secret_access_key=aws_secret,
        aws_session_token=aws_token,
        region_name=REGION
    )

    local_path = os.path.join(tempfile.gettempdir(), MODEL_INFO["explainer"])

    if not os.path.exists(local_path):
        s3.download_file(
            Bucket=aws_bucket,
            Key=f"explainer/{MODEL_INFO['explainer']}",
            Filename=local_path
        )

    return joblib.load(local_path)

# ----------------------------
# Call SageMaker endpoint
# ----------------------------
def call_model_api(input_df):
    try:
        payload = {"data": input_df.to_dict(orient="records")}

        response = runtime.invoke_endpoint(
            EndpointName=MODEL_INFO["endpoint"],
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(payload)
        )

        result = json.loads(response["Body"].read().decode("utf-8"))
        prediction = float(result["predictions"][-1])
        return round(prediction, 4), 200

    except Exception as e:
        return str(e), 500

# ----------------------------
# SHAP Visualization
# ----------------------------
def display_explanation(input_df):
    explainer = load_shap()
    shap_values = explainer(input_df)

    st.subheader("🔍 Decision Transparency (SHAP)")

    plt.figure(figsize=(10,4))
    shap.plots.waterfall(shap_values[-1], max_display=10, show=False)
    st.pyplot(plt.gcf())
    plt.close()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="ML Deployment", layout="wide")
st.title("👨‍💻 ML Deployment")

with st.form("pred_form"):
    cols = st.columns(2)
    user_inputs = {}

    for i, inp in enumerate(MODEL_INFO["inputs"]):
        with cols[i % 2]:
            user_inputs[inp["name"]] = st.number_input(
                inp["name"],
                min_value=float(inp["min"]),
                max_value=float(inp["max"]),
                value=float(inp["default"]),
                step=float(inp["step"])
            )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    row = [user_inputs[k] for k in MODEL_INFO["keys"]]
    input_df = pd.concat(
        [df_features, pd.DataFrame([row], columns=df_features.columns)],
        ignore_index=True
    )

    res, status = call_model_api(input_df)

    if status == 200:
        st.metric("Prediction Result", res)
        display_explanation(input_df)
    else:
        st.error(res)
