import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import pandas as pd

# -----------------------------
# Config
# -----------------------------

API_URL = "https://healthai-backend-oj25.onrender.com"  # FastAPI backend base URL

st.set_page_config(
    page_title="HealthAI - X-ray Analysis",
    page_icon="🩻",
    layout="centered"
)

# -----------------------------
# Helper functions (UX logic)
# -----------------------------

def get_confidence_level(prob: float) -> str:
    """
    Simple confidence interpretation for a probability between 0 and 1.
    We treat distance from 0.5 as confidence.
    """
    # distance from uncertain mid-point (0.5)
    dist = abs(prob - 0.5)

    if dist >= 0.3:
        return "High"
    elif dist >= 0.15:
        return "Medium"
    else:
        return "Low"


def format_confidence_message(label: str, prob: float) -> str:
    """
    Generate a human-readable string summarizing prediction & confidence.
    """
    conf = get_confidence_level(prob)
    if label.upper() == "PNEUMONIA":
        base = f"Model predicts **PNEUMONIA** with **{prob:.2f}** probability."
    else:
        base = f"Model predicts **NORMAL** with **{1 - prob:.2f}** probability (Pneumonia prob: {prob:.2f})."

    return f"{base}  \n**Confidence level:** {conf}."


# -----------------------------
# UI
# -----------------------------

st.title("🩻 HealthAI - Chest X-ray Analysis")

st.write(
    """
    HealthAI can analyze chest X-ray images using:
    
    - **Simple model**: Pneumonia vs Normal  
    - **Advanced model (CheXpert)**: Multi-disease probability prediction  
    """
)

mode = st.radio(
    "Select analysis mode:",
    ["Pneumonia vs Normal (Simple)", "Multi-disease (CheXpert)"]
)

# Multi-disease threshold slider (only relevant for CheXpert mode)
if mode.startswith("Multi-disease"):
    threshold = st.slider(
        "🔎 Minimum probability to **flag** a condition",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        help="Conditions with predicted probability above this threshold will be highlighted."
    )
else:
    threshold = None  # not used in simple mode

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Show preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    if st.button("🔍 Analyze X-ray"):
        with st.spinner("Sending image to HealthAI backend..."):
            try:
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }

                # ------------------ SIMPLE MODE ------------------
                if mode.startswith("Pneumonia"):
                    resp = requests.post(f"{API_URL}/predict-xray", files=files)
                    if resp.status_code == 200:
                        data = resp.json()
                        label = data.get("predicted_label", "Unknown")
                        pneu_prob = float(data.get("pneumonia_probability", 0.0))
                        normal_prob = 1.0 - pneu_prob

                        st.subheader("🩺 Simple Pneumonia Prediction")

                        # Show probabilities in a small table
                        prob_df = pd.DataFrame(
                            [
                                {"Class": "Pneumonia", "Probability": pneu_prob},
                                {"Class": "Normal", "Probability": normal_prob},
                            ]
                        )
                        st.table(prob_df.style.format({"Probability": "{:.2f}"}))

                        # Confidence message
                        st.write(format_confidence_message(label, pneu_prob))

                        # Alert style messages
                        if label.upper() == "PNEUMONIA":
                            st.warning(
                                "⚠️ The model leans towards **PNEUMONIA**. "
                                "This is an AI assistant prediction, not a medical diagnosis. "
                                "Please consult a medical professional for confirmation."
                            )
                        else:
                            st.info(
                                "✅ The model leans towards **NORMAL**. "
                                "However, AI predictions are not a substitute for clinical evaluation."
                            )
                    else:
                        st.error(f"Backend error ({resp.status_code}): {resp.text}")

                # ------------------ MULTI-DISEASE MODE ------------------
                else:
                    resp = requests.post(f"{API_URL}/predict-xray-multidisease", files=files)
                    if resp.status_code == 200:
                        data = resp.json()
                        predictions = data.get("predictions", {})
                        top3 = data.get("top3", [])

                        st.subheader("🧬 Multi-disease Probabilities (CheXpert)")

                        if predictions:
                            # Convert predictions to DataFrame
                            df = pd.DataFrame(
                                [
                                    {"Condition": k, "Probability": v}
                                    for k, v in predictions.items()
                                ]
                            ).sort_values("Probability", ascending=False)

                            # Show full table
                            st.markdown("**All predicted conditions (sorted):**")
                            st.dataframe(df.style.format({"Probability": "{:.2f}"}))

                            # Highlight flagged conditions above threshold
                            if threshold is not None:
                                flagged = df[df["Probability"] >= threshold]
                                if not flagged.empty:
                                    st.markdown(f"### 🚩 Conditions above threshold ({threshold:.2f})")
                                    st.table(flagged.style.format({"Probability": "{:.2f}"}))
                                else:
                                    st.markdown(
                                        f"✅ No conditions exceeded the threshold of **{threshold:.2f}**."
                                    )

                            # Top 3 section
                            if top3:
                                st.markdown("### 🏆 Top 3 predicted conditions")
                                for item in top3:
                                    st.write(
                                        f"- **{item['label']}** → probability: **{item['probability']:.2f}**"
                                    )

                            st.info(
                                "📝 These probabilities are **model estimates** based on patterns in X-ray images. "
                                "They **must not** be used as a standalone medical diagnosis. "
                                "Always consult a qualified healthcare professional."
                            )
                        else:
                            st.warning("No predictions returned by backend.")

                    else:
                        st.error(f"Backend error ({resp.status_code}): {resp.text}")

            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
else:
    st.info("Please upload a chest X-ray image to begin.")
