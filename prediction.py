import streamlit as st
import pandas as pd
import joblib

def prediction_app():
    st.title("💳 Prediksi Risiko Fraud Kartu Kredit🔍")
    st.write("Masukkan detail transaksi untuk memprediksi risiko fraud menggunakan model Logistic Regression.")

    # 1) Load Model & Metadata (hasil training fraud kamu)
    model = joblib.load("model_fraud.pkl")
    feature_names = joblib.load("model_features.pkl")       # daftar kolom setelah get_dummies (training)
    numeric_cols = joblib.load("numeric_columns.pkl")       # kolom numerik asli yang di-scaling saat training
    scaler = joblib.load("scaler_fraud.pkl")                      # scaler hasil training (WAJIB)

    st.write("### Input Data Transaksi")
    # --- Top 10 pilihan untuk dropdown (bisa kamu sesuaikan dengan dataset kamu) ---
    TOP_10_COUNTRIES = [
    "USA", "UK", "Canada", "Germany", "France",
    "Australia", "Japan", "India", "Brazil", "Indonesia"
    ]

    TOP_10_CATEGORIES = [
    "Grocery", "Fuel", "Online Services", "Entertainment", "Food",
    "Electronics", "Travel", "Clothing", "Health", "Others"
    ]


    # 2) Form Input Pengguna (sesuai kolom dataset fraud kamu)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        amount = st.number_input("Amount (USD)", min_value=0.0, value=120.0, step=1.0)
    with col2:
        distance = st.number_input("Distance From Home (km)", min_value=0.0, value=5.0, step=0.5)
    with col3:
        hour = st.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=14, step=1)
    with col4:
        is_international = st.selectbox("International?", ["No", "Yes"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        device_type = st.selectbox("Device Type", ["Mobile", "Terminal", "Web"])
    with col2:
        card_type = st.selectbox(
    "Card Type",
    ["Credit", "Debit", "Platinum", "Gold"])   
    with col3:
        transaction_type = st.selectbox("Transaction Type", ["ATM", "POS", "Online"])
    with col4:
        country = st.selectbox("Country", TOP_10_COUNTRIES, index=0)


    col1, col2, col3,col4 = st.columns(4)
    with col1:
        merchant_category = st.selectbox("Merchant Category", TOP_10_CATEGORIES, index=0)
    with col2:
        is_chip = st.selectbox("Chip Used?", ["No", "Yes"])
    with col3:
        is_pin = st.selectbox("PIN Used?", ["No", "Yes"])
    with col4:
        day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])   

    # Konversi ke format 0/1 yang dipakai dataset
    is_international_val = 1 if is_international == "Yes" else 0
    is_chip_val = 1 if is_chip == "Yes" else 0
    is_pin_val = 1 if is_pin == "Yes" else 0

    # 3) Kategori Otomatis 
    st.write("")
    cA, cB, cC, cD = st.columns(4)

    # Amount Category
    with cA:
        if amount < 50:
            amount_cat = "Low"
            color = "#4CAF50"
        elif amount < 200:
            amount_cat = "Medium"
            color = "#EBD300"
        else:
            amount_cat = "High"
            color = "#F44336"
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">Amount Category</div>
                <div style="font-size:16px;font-weight:bold;">{amount_cat}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Distance Category
    with cB:
        if distance <= 1:
            dist_cat = "Near"
            color = "#4CAF50"
        elif distance <= 10:
            dist_cat = "Moderate"
            color = "#EBD300"
        else:
            dist_cat = "Far"
            color = "#F44336"
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">Distance Category</div>
                <div style="font-size:16px;font-weight:bold;">{dist_cat}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Time Category
    with cC:
        if 5 <= hour <= 10:
            time_cat = "Morning"
            color = "#4CAF50"
        elif 11 <= hour <= 14:
            time_cat = "Noon"
            color = "#EBD300"
        elif 15 <= hour <= 18:
            time_cat = "Afternoon"
            color = "#EBD300"
        else:
            time_cat = "Night"
            color = "#F44336"
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">Time Category</div>
                <div style="font-size:16px;font-weight:bold;">{time_cat}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Security Level (Chip + PIN)
    with cD:
        if (is_chip_val == 1) and (is_pin_val == 1):
            sec_cat = "Strong"
            color = "#4CAF50"
        elif (is_chip_val == 1) or (is_pin_val == 1):
            sec_cat = "Medium"
            color = "#EBD300"
        else:
            sec_cat = "Weak"
            color = "#F44336"
        st.markdown(
            f"""
            <div style="background:{color}; padding:10px; border-radius:10px; text-align:center; color:#fff;">
                <div style="font-size:13px;">Security Level</div>
                <div style="font-size:16px;font-weight:bold;">{sec_cat}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")

    # 4) Masukkan menjadi DataFrame (kolom mentah sesuai training sebelum get_dummies)
    user_df = pd.DataFrame({
        "Amount": [amount],
        "Distance_From_Home": [distance],
        "Hour_of_Day": [hour],
        "Device_Type": [device_type],
        "Card_Type": [card_type],
        "Transaction_Type": [transaction_type],
        "Merchant_Category": [merchant_category],
        "Country": [country],
        "Is_International": [is_international_val],
        "Is_Chip": [is_chip_val],
        "Is_Pin_Used": [is_pin_val],
    })

    # 5) One-Hot Encoding (harus sama dengan training)
    user_processed = pd.get_dummies(user_df, drop_first=True)

    # Tambahkan kolom yang hilang agar sama persis dengan training
    for col in feature_names:
        if col not in user_processed.columns:
            user_processed[col] = 0

    # Buang kolom extra yang tidak ada di training (kalau ada)
    user_processed = user_processed[feature_names]

    # 6) Scaling numerik pakai scaler hasil training (WAJIB transform, bukan fit_transform)
    if scaler is not None and len(numeric_cols) > 0:
        cols_to_scale = [c for c in numeric_cols if c in user_processed.columns]
        user_processed[cols_to_scale] = scaler.transform(user_processed[cols_to_scale])

# 7) Prediksi
    if st.button("Prediksi Fraud"):
        prob = model.predict_proba(user_processed)[0][1]

        # threshold 
        threshold = 0.40   
        pred = 1 if prob >= threshold else 0

        st.write("### 🔍 Hasil Prediksi")
        st.metric("Probabilitas Fraud", f"{prob*100:.2f}%")

        # Risk level (tetap pakai level seperti punyamu)
        if prob >= 0.70:
            st.error("⚠️ Transaksi berisiko **TINGGI** terindikasi **FRAUD**.")
            risk_level = "High Risk"
        elif prob >= 0.40:
            st.warning("⚠️ Transaksi berisiko **SEDANG**. Perlu verifikasi tambahan.")
            risk_level = "Medium Risk"
        else:
            st.success("✅ Transaksi berisiko **RENDAH** (cenderung Non-Fraud).")
            risk_level = "Low Risk"

        st.write("---")
        st.write("### 📌 Interpretasi")
        st.write(f"- **Prediction (0/1)**: {pred}")
        st.write(f"- **Risk Level**: {risk_level}")
        st.markdown("""
Model memprediksi risiko fraud berdasarkan pola historis transaksi serupa 
yang pernah terjadi sebelumnya. Nilai probabilitas menunjukkan seberapa besar 
kemiripan transaksi ini dengan pola fraud yang pernah teridentifikasi.

Perlu diperhatikan bahwa hasil prediksi **bukan keputusan final**, melainkan 
alat bantu untuk mendukung proses verifikasi. Dalam praktik bisnis, hasil ini 
sebaiknya dikombinasikan dengan kebijakan internal, aturan keamanan tambahan, 
dan validasi manual pada transaksi berisiko tinggi.
""")

# panggilan fungsi ini untuk jalankan aplikasi prediksi
if __name__ == "__main__":
    prediction_app()

