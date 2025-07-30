import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Benchmarking", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# ---------------------- Load Models --------------------------

with open('forecast_multioutput_models.pkl', 'rb') as f:
    model_data = pickle.load(f)

models = model_data['models']
scaler = model_data['scaler']
accuracies = model_data['accuracies']

target_labels_all = [
    'Total Power Cable',
    'LCS',
    'Total Light Fixture',
    'Total Cable Tray',
    'Total Earthing Material (GI Strip)',
    'Total Earthing Material (Equipment Earthing)',
]

# ---------------------- Sidebar --------------------------

with st.sidebar:
    st.image("/Users/adityadubey/Desktop/Benchmarking_Project/Image.png", width=150)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:23px; font-weight: bold; color: #FFF1D5">📊 Select Forecasting Model</p>', unsafe_allow_html=True)
    model_choice = st.selectbox(
        "",
        list(models.keys()),
        label_visibility="collapsed"
    )

    st.markdown(
    f"""
    <div style="
        background-color: #f0f8ff;
        padding: 10px 15px;
        border-radius: 10px;
        margin-top: 10px;
        font-size: 16px;
        font-weight: bold;
        color: #0a4d68;
        border: 1px solid #d1e7f7;
        ">
        📈 Model Accuracy (R² Score): <span style="color:#005580;">{accuracies[model_choice]:.2f}</span>
    </div>
    """,
    unsafe_allow_html=True
    )

# ---------------------- Main Section --------------------------

st.markdown("<h1 class='title'>🔧 Electrical Bulk Material Benchmarking</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict key designing parameters based on your project inputs</p>", unsafe_allow_html=True)

# Input section
with st.container():
    st.markdown("## 🔢 Enter your project inputs")
    col1, col2, col3 = st.columns(3)

    with col1:
        feature1 = st.number_input("📐 Plot size (sq m)", value=0.0, help="Total land area of the project.")

    with col2:
        feature2 = st.number_input("👥 Total consumer", value=0.0, help="Number of consumers/sub-consumers to be served.")

    with col3:
        feature3 = st.number_input("⚙️ Total Equipment (Mechanical + Electrical)", value=0.0, help="Total count of installed equipment.")

input_df = pd.DataFrame([[feature1, feature2, feature3]], columns=[
    'Plot size (sq m)', 'Total consumer', 'Total Equipment (Mechanical + Electrical)'
])

# Predict Button
if st.button("🔮 Run Forecast"):
    st.markdown("## 📈 Forecasted Outputs")

    if feature3 == 0.0:
        st.warning("⚠️ Third input (Total Equipment (Mechanical + Electrical) is missing. Only first 4 outputs will be predicted.")
        temp_input_df = input_df.copy()
        temp_input_df.iloc[0, 2] = input_df.iloc[:, 2].mean()

        if model_choice == 'Ridge':
            input_scaled = scaler.transform(temp_input_df)
            prediction = models[model_choice].predict(input_scaled)
        else:
            prediction = models[model_choice].predict(temp_input_df)

        prediction = prediction[0]

        col1, col2 = st.columns(2)
        for i in range(4):
            with [col1, col2][i % 2]:
                st.markdown(f"""
                    <div class="metric-box">
                        ✅ {target_labels_all[i]} -  
                        <span style='font-size:20px;color:white;'><b>{prediction[i]:.2f}</b></span>
                    </div>
                """, unsafe_allow_html=True)

        st.error("❌ Last 2 outputs skipped due to missing 'Total Equipment (Mechanical + Electrical)' input.")
    else:
        if model_choice == 'Ridge':
            input_scaled = scaler.transform(input_df)
            prediction = models[model_choice].predict(input_scaled)
        else:
            prediction = models[model_choice].predict(input_df)

        prediction = prediction[0]

        col1, col2 = st.columns(2)
        for i in range(len(target_labels_all)):
            with [col1, col2][i % 2]:
                st.markdown(f"""
                    <div class="metric-box">
                        ✅ {target_labels_all[i]} -  
                        <span style='font-size:20px;color:white;'><b>{prediction[i]:.2f}</b></span>
                    </div>
                """, unsafe_allow_html=True)

# ---------------------- Foooter --------------------------

st.markdown("---")
st.markdown(
    """
    <div style='display: flex; justify-content: space-between; align-items: center; color: white;'>
        <p style='margin: 0;'>© 2025 Electrical Bulk Material Benchmarking. All rights reserved.</p>
        <a href='https://www.ten.com/en' target='_blank' style='color: white; text-decoration: none;'>🌐 Visit Technip Energies</a>
    </div>
    """,
    unsafe_allow_html=True
)
