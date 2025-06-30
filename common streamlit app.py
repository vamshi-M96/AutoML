import streamlit as st
import pandas as pd
import pickle

st.title("🔮 Simple ML Predictor")

model_file = st.file_uploader("Upload Model (.pkl)", type=["pkl"])
data_file = st.file_uploader("Upload Training Data (.csv or .xlsx)", type=["csv", "xlsx"])
     # ✅ correct
    #st.write(model_file.feature_names_in_)
if model_file and data_file:
    try:
        model_file.seek(0)
        model = pickle.load(model_file)
        features = list(model.feature_names_in_)
    except Exception as e:
        st.error(f"🚫 Model load error: {e}")
        st.stop()

    try:
        data_file.seek(0)
        df = pd.read_csv(data_file) if data_file.name.endswith(".csv") else pd.read_excel(data_file)
    except Exception as e:
        st.error(f"🚫 Data load error: {e}")
        st.stop()

    # ✅ Identify target column (first non-feature column)
    target_candidates = [col for col in df.columns if col not in features]
    target = target_candidates[0] if target_candidates else "Target"

    st.info(f"🎯 The model is predicting: **{target}**")

    st.subheader("📥 Input Feature Values")
    input_data = {}
    cols = st.columns(3)

    for i, col in enumerate(features):
        # ✅ Safely get min and max if column exists
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            min_, max_ = df[col].min(), df[col].max()
        else:
            min_, max_ = 0, 100  # Fallback range

        with cols[i % 3]:
            val = st.text_input(f"{col} ({min_}–{max_})", key=col)
            try:
                input_data[col] = float(val)
            except:
                st.warning(f"Invalid input for {col}")
                st.stop()

    if st.button("Predict"):
        try:
            input_df = pd.DataFrame([input_data])
            pred = model.predict(input_df)
            st.success(f"✅ Predicted {target}: {pred[0]}")
        except Exception as e:
            st.error(f"🚫 Prediction error: {e}")