import numbers
import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE
import joblib


def ml_model():
    df = pd.read_csv("credit_card_fraud_10k_5877fraud_4123nonfraud.csv")

    st.write("## 🤖 Machine Learning — Fraud Detection (Logistic Regression)")

        # 1. menampilkan 5 dataset teratas
    st.write("### 📊 Preview Datasets")
    st.dataframe(df.head())

    # 2. menghapus kolom yang tidak diperlukan
    df = df.drop(columns=['Transaction_Date'])

    # 3. membagi kolom numerik dan kategorik
    numbers = df.select_dtypes(include=['number']).columns
    categories = df.select_dtypes(exclude=['number']).columns

    # 4. Deteksi outlier dengan IQR (HANYA INFO, TIDAK DIHAPUS)
    st.write("### 1. Deteksi Outlier ")

    total_rows = df.shape[0]
    st.write(f"Jumlah data: **{total_rows:,} baris**")

    if len(numbers) == 0:
        st.info("Tidak ada kolom numerik untuk deteksi outlier.")
    else:
        Q1 = df[numbers].quantile(0.25)
        Q3 = df[numbers].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # True jika sebuah nilai outlier pada kolom numerik
        outlier_flags = (df[numbers] < lower_bound) | (df[numbers] > upper_bound)

        # Hitung baris yang mengandung minimal 1 outlier
        outlier_rows = int(outlier_flags.any(axis=1).sum())
        outlier_pct = (outlier_rows / total_rows) * 100 if total_rows > 0 else 0

        st.write(f"Jumlah Outlier: **{outlier_rows:,} baris** ({outlier_pct:.2f}%)")

        st.caption(
            "Catatan: Outlier **tidak dihapus** karena pada kasus fraud detection, transaksi fraud sering muncul "
            "sebagai nilai ekstrem (misalnya amount sangat besar atau jarak sangat jauh). Jika outlier dihapus, "
            "kita berisiko menghapus pola fraud yang penting sehingga recall untuk fraud bisa turun."
        )

        # 5. One-Hot Encoding untuk kolom kategorik
    df_select = df.copy()

    # pastikan target tidak ikut di-encode
    categories = df_select.select_dtypes(exclude=["number"]).columns.tolist()
    categories = [c for c in categories if c != "Fraud_Flag"]

    # One-Hot Encoding
    df_select = pd.get_dummies(df_select, columns=categories, drop_first=True)

  

    # 6. normalisasi dengan minmaxscaler
    st.write("### 2. Normalisasi menggunakan Min-Max Scaler")
    df_scaled_vis = df_select.copy()
    for col in numbers:
        if col in df_scaled_vis.columns and col != "Fraud_Flag":
            df_scaled_vis[col] = MinMaxScaler().fit_transform(df_scaled_vis[[col]])

    # 7. korelasi heatmap
    st.write("### 3. Korelasi Linear antar Kolom Numerik")
    col1,col2 = st.columns([6,4])
    with col1:
        # 7.a Heatmap Korelasi
        corr = df_select[numbers].corr().round(2)
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 7.b Deskripsi Korelasi Heatmap
        st.write(" Deskripsi Correlation Heatmap ")
        st.write(""" 
        - Sebagian besar fitur punya korelasi rendah (mendekati 0), jadi informasi antar fitur relatif tidak saling “duplikat”.
        - Ada beberapa hubungan beberapa kolom seperti : Amount_Diff dan Amount sangat tinggi (≈ 0,91), 
            Fraud_Flag dan Is_Night cukup kuat negatif (≈ -0,75), dan No_Pin_Chip negatif dengan Is_Chip/Is_Pin_Used (≈ -0,74/-0,56).
        - Karena Amount_Diff membawa informasi yang hampir sama dengan Amount (berpotensi multikolinearitas), kolom Amount_Diff dihapus dan Amount dipertahankan.
        """)

    # Drop Amount_Diff sebelum Train-Test Split
    df_select = df_select.drop(columns=["Amount_Diff"], errors="ignore")

    #Train Test Split
    st.write("### 4. Train Test Split")
    x = df_select.drop('Fraud_Flag', axis=1)
    y = df_select['Fraud_Flag']
    #Membagi data menjadi 80% untuk training dan 20% untuk testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,stratify=y)
    st.write(f"Jumlah data training: {x_train.shape[0]} baris")
    st.write(f"Jumlah data testing: {x_test.shape[0]} baris")   

    # rapikan list numerik
    numbers = [c for c in numbers if c not in ["Fraud_Flag", "Amount_Diff", "amount_diff"]]
    numbers = [c for c in numbers if c in x_train.columns]

    # FIT scaler di TRAIN, lalu transform TRAIN & TEST
    scaler = MinMaxScaler()
    x_train[numbers] = scaler.fit_transform(x_train[numbers])
    x_test[numbers]  = scaler.transform(x_test[numbers])

    #Handle Imbalanced Data dengan SMOTE
    st.write("### 5. Penanganan Imbalanced Data ")  
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Mengapa Imbalanced Data Perlu Ditangani?**")
        st.write("Model Bias: Jika satu kelas jauh lebih banyak (misalnya Non-Fraud), model cenderung “bermain aman” dengan lebih sering memprediksi kelas mayoritas, sehingga kasus penting pada kelas minoritas (Fraud) mudah terlewat.")
        st.write("Metrik seperti akurasi bisa tampak tinggi karena model cukup menebak kelas mayoritas, namun kemampuan mendeteksi kelas minoritas (yang justru penting) rendah.")
        st.write("Generalisasi Terbatas: Model tidak belajar pola yang kuat untuk kelas minoritas, sehingga saat bertemu data baru, performa untuk mendeteksi fraud bisa buruk dan meningkatkan risiko False Negative (fraud lolos)")
    with col2:
        st.write('**Sebelum Balancing**')
        col1,col2 = st.columns(2)
        with col1:
            total_0 = (y_train == 0).sum()
            st.metric(label='Label 0', value=total_0)
        with col2:
            total_1 = (y_train == 1).sum()
            st.metric(label='Label 1', value=total_1)
        st.write('**Setelah Balancing**')
        sm = SMOTE(random_state=42)
        x_train_balance, y_train_balance = sm.fit_resample(x_train, y_train)    
        col1,col2 = st.columns(2)
        with col1:
            st.metric(label='Label 0', value=(y_train_balance == 0).sum())
        with col2:
            st.metric(label='Label 1', value=(y_train_balance == 1).sum())
    
    #9. Training Model dengan Logistic Regression
    st.write('### 6. Pemodelan')
    model = LogisticRegression()
    model.fit(x_train, y_train)
    train_accuracy = model.score(x_train, y_train)
    st.write("Akurasi Training =", round(train_accuracy * 100, 2), "%")

    col1, col2 = st.columns([6,4])
    with col1:
        st.write('**Parameter Model Logistic Regression**')
        feature_names = x_train.columns
        beta_0 = model.intercept_[0]           # Intercept
        beta = model.coef_[0]                  # Koefisien fitur

        st.write("**β0 (Intercept)**")
        st.write(beta_0)

        st.write("**β1, β2, ..., βn (Koefisien per Feature)**")
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient (β)": beta}).sort_values(by="Coefficient (β)", ascending=False)    
        st.dataframe(coef_df)
    
    with col2:
        st.write('**Interpretasi Model**')
        st.write("**a. Koefisien Positif → Meningkatkan Peluang Fraud**")
        st.write("Semakin besar nilai β, maka ketika fitur tersebut meningkat/aktif " \
                    "peluang transaksi diprediksi Fraud (1) cenderung naik.")
        
        st.write("**b. Koefisien Negatif → Menurunkan Peluang Fraud**")
        st.write("Jika β bernilai negatif, maka ketika fitur tersebut meningkat/aktif," \
                    "peluang transaksi diprediksi Fraud (1) cenderung turun.")
        
        st.write('**c. Intercept Model (β0)**')
        st.markdown(f"""
            Nilai β0 = **{beta_0:.4f}**  
            menunjukkan bahwa peluang dasar (ketika semua variabel independen 
            tidak ada atau diatur ke nol) peluang Fraud sangat rendah..""")
        
    st.write('**d. Faktor Risiko Tertinggi (β positif terbesar)**')
    st.markdown(f"""
        Dari hasil model, fitur yang paling mendorong prediksi Fraud adalah:
        - **High_Deviation** transaksi dengan pola nilai yang “anomali/tidak wajar” paling kuat terkait Fraud.
        - **Merchant_Category_Online Services** serta kategori **Fuel/Food/Entertainment/Groceries** → beberapa kategori merchant lebih sering muncul pada transaksi Fraud dibanding kategori baseline.
        - **Risk_Score.** → makin tinggi skor risiko, makin tinggi peluang terdeteksi Fraud.
        - **No_Pin_Chip** → transaksi yang *tidak memenuhi kombinasi keamanan Chip+PIN* cenderung lebih berisiko.
        """)
        
    st.write('**e. Faktor yang Menurunkan Peluang Fraud (β negatif terbesar)**')
    st.markdown("""
    Dari hasil koefisien negatif, fitur yang cenderung menurunkan prediksi Fraud adalah:
    - **Amount** →  transaksi dengan amount lebih besar cenderung *lebih sering* diprediksi non-fraud (kebalikannya: fraud banyak terjadi pada nominal tertentu/lebih kecil).
    - **Freq_Transactions** → semakin sering transaksi , model menangkap pola yang lebih “normal”.
    - **Distance_From_Home** dan beberapa **Country/Day_of_Week** → efeknya relatif lebih kecil, dan interpretasinya tergantung kategori.
        """) 

    #10 Evaluasi Model
    st.write('### 7. Evaluasi Model')
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision= precision_score(y_test, y_pred)
    recall= recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(x_test)[:,1])

    col1, col2 = st.columns([6,4])
    with col1:
        cm = confusion_matrix(y_test, y_pred)
        labels = model.classes_
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        fig = px.imshow(cm_df,text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"), aspect="auto")
        fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Label",
            yaxis_title="Actual Label", height=400, width=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Akurasi", value=f"{round(accuracy * 100, 2)}%")
            st.metric(label="Precision", value=f"{round(precision * 100, 2)}%")
            st.metric(label="Recall", value=f"{round(recall * 100, 2)}%")
        with col2:
            st.metric(label="F1 Score", value=f"{round(f1 * 100, 2)}%")
            st.metric(label="ROC AUC", value=f"{round(roc_auc * 100, 2)}%")
# 11. Insight dari Hasil Pemodelan
    st.write("### 8. Evaluasi Hasil Pemodelan")

    col1, col2 = st.columns(2)

# Ambil angka confusion matrix
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    with col1:
        st.write("**Kualitas Prediksi Keseluruhan**")
        st.markdown(f"""
    Berdasarkan hasil evaluasi, dapat disimpulkan bahwa model:
    - **Akurasi: {accuracy*100:.2f}%** → sekitar {accuracy*100:.2f}% prediksi model sudah benar secara keseluruhan.
    - **F1 Score: {f1*100:.2f}%** → cukup baik karena model sangat kuat menangkap Fraud (recall tinggi), walau masih banyak false alarm.
    - **ROC AUC: {roc_auc*100:.2f}%** → mendekati **50%**, Model lebih dioptimalkan untuk recall fraud daripada separasi probabilistik global
        """)

        st.write("**Fokus pada Kelas Positif (Fraud = 1)**")
        st.write("Dalam fraud detection, target utama adalah meminimalkan **False Negative (FN)**, yaitu transaksi fraud yang lolos terdeteksi.")

        st.markdown(f"""
    - **Recall (Fraud): {recall*100:.2f}%** → dari semua fraud yang benar-benar terjadi, model berhasil menangkap hampir semuanya.  
      Ini sangat baik karena **FN hanya {FN}** (fraud yang lolos sangat sedikit).
    - **Precision: {precision*100:.2f}%** → dari semua transaksi yang diprediksi Fraud, hanya {precision*100:.2f}% yang benar-benar fraud.  
      Ini menunjukkan **False Positive masih tinggi (FP = {FP})**, jadi banyak transaksi normal ikut “ditandai fraud”.
     """)

    with col2:
        st.write("**Rekomendasi Peningkatan Kualitas Model**")

        st.write("**a. Kurangi False Positive dengan Threshold**")
        st.write(
        "Model saat ini kemungkinan memakai threshold default 0.5. "
        "Karena FP tinggi, coba naikkan threshold (misal 0.6–0.8) supaya model lebih selektif "
        "dan tidak terlalu banyak menandai transaksi normal sebagai fraud."
        )

        st.write("**b. Sesuaikan tujuan bisnis**")
        st.write("""
        - Prioritas utama adalah **mencegah fraud lolos**, model ini sudah bagus (FN sangat kecil).
        - Kalau sistem sensitif terhadap “false alarm” (transaksi normal keblokir), maka fokus perbaikan adalah **menurunkan FP**.
        """)
    
    import joblib

    feature_names = x_train.columns.tolist()

    joblib.dump(model, "model_fraud.pkl")
    joblib.dump(feature_names, "model_features.pkl")
    joblib.dump(list(numbers), "numeric_columns.pkl")
    joblib.dump(scaler, "scaler_fraud.pkl")


    

