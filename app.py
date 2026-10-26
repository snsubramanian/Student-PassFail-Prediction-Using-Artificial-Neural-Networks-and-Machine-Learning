import streamlit as st
import pandas as pd
import joblib

# Load saved model
bundle = joblib.load("student_pass_fail_model.pkl")

model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

st.set_page_config(
    page_title="Student Pass/Fail Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Pass/Fail Prediction")
st.write("Enter the student's academic and attendance details.")

st.info(
    "This app uses a Random Forest model trained on your dataset. "
    "Predictions are estimates, not guaranteed outcomes."
)

defaults = bundle["defaults"]
minimums = bundle["minimums"]
maximums = bundle["maximums"]

input_data = {}

with st.form("student_form"):
    st.subheader("Student Details")

    for feature in feature_names:
        min_val = float(minimums[feature])
        max_val = float(maximums[feature])
        default_val = float(defaults[feature])

        # Binary encoded features, such as gender_Male
        if min_val >= 0 and max_val <= 1:
            input_data[feature] = st.selectbox(
                feature.replace("_", " ").title(),
                options=[0, 1],
                index=int(round(default_val))
            )
        else:
            # Give constant-valued features a small valid range
            if min_val == max_val:
                min_val -= 1
                max_val += 1

            default_val = max(min_val, min(default_val, max_val))

            input_data[feature] = st.number_input(
                feature.replace("_", " ").title(),
                min_value=min_val,
                max_value=max_val,
                value=default_val
            )

    submitted = st.form_submit_button("Predict Result")

if submitted:
    input_df = pd.DataFrame([input_data], columns=feature_names)

    # Apply the same scaling used during training
    input_scaled = scaler.transform(input_df)

    prediction = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = float(probabilities[prediction]) * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Predicted Result: PASS")
    else:
        st.error("Predicted Result: FAIL")

    st.write(f"Model confidence for this prediction: {confidence:.2f}%")
    st.caption(
        "This confidence is the model's estimated probability, "
        "not a guarantee that the prediction is correct."
    )