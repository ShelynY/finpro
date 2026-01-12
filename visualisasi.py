from altair import sample
import streamlit as st
import pandas as pd
import plotly.express as px

def chart():

    df = pd.read_csv('credit_card_fraud_final_2025.csv')

    # normalisasi device type supaya button selalu match
    df["Device_Type"] = df["Device_Type"].astype(str).str.strip().str.title()

    if "selected_device" not in st.session_state:
        st.session_state.selected_device = None


# KPI ROW 

    st.markdown("""
    <style>
    /* Perbesar tampilan st.metric */
    div[data-testid="stMetric"]{
  background: linear-gradient(135deg, rgba(99,102,241,.10), rgba(56,189,248,.10));
  border: 1px solid rgba(15,23,42,.10);
  padding: 18px 18px;
  border-radius: 16px;
  box-shadow: 0 8px 18px rgba(15,23,42,.08);
    }
    div[data-testid="stMetricLabel"] > div{
  font-size: 15px !important;
  color: rgba(15,23,42,.75) !important;
  font-weight: 700 !important;
    }
    div[data-testid="stMetricValue"]{
  font-size: 34px !important;
  color: rgba(15,23,42,.92) !important;
  font-weight: 900 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col4, col5 = st.columns([2, 2, 3, 2])

# Default KPI (sebelum filter) -> nanti akan ditimpa setelah apply filter
    total_transaction = df.shape[0]
    fraud_count = int(df["Fraud_Flag"].sum())
    amount_lost_to_fraud = df.loc[df["Fraud_Flag"] == 1, "Amount"].sum()
    avg_loss_per_fraud = (amount_lost_to_fraud / fraud_count) if fraud_count > 0 else 0

    with col1:
        st.metric("Total Transaksi", f"{total_transaction:,}")
    with col2:
        st.metric("Transaksi Fraud", f"{fraud_count:,}")
    with col4:
        st.metric("Amount Lost to Fraud (USD)", f"${amount_lost_to_fraud:,.2f}")
    with col5:
        st.metric("Average Loss per Fraud (USD)", f"${avg_loss_per_fraud:,.2f}")

# FILTER ROW 

    st.write("**Device Type**")
    b1, b2, b3, b4, b5 = st.columns([1, 1, 1, 1.4, 0.5])

    with b1:
        if st.button("Mobile", key="device_mobile"):
            st.session_state.selected_device = "Mobile"
    with b2:
        if st.button("Terminal", key="device_terminal"):
            st.session_state.selected_device = "Terminal"
    with b3:
        if st.button("Web", key="device_web"):
            st.session_state.selected_device = "Web"
    with b4:
        if st.button("All Devices", key="device_all"):
            st.session_state.selected_device = None
    with b5:
        if st.button("🔄", key="reset_device"):
            st.session_state.selected_device = None
            st.rerun()

# APPLY FILTER

    df_filtered = df.copy()
    if st.session_state.selected_device is not None:
        df_filtered = df_filtered[df_filtered["Device_Type"] == st.session_state.selected_device]



    total_transaction_f = df_filtered.shape[0]
    fraud_count_f = int(df_filtered["Fraud_Flag"].sum())
    amount_lost_to_fraud_f = df_filtered.loc[df_filtered["Fraud_Flag"] == 1, "Amount"].sum()
    avg_loss_per_fraud_f = (amount_lost_to_fraud_f / fraud_count_f) if fraud_count_f > 0 else 0

    col1f, col2f, col4f, col5f = st.columns([2, 2, 3, 2])
    with col1f:
        st.metric("Total Transaksi (Filtered)", f"{total_transaction_f:,}")
    with col2f:
        st.metric("Transaksi Fraud (Filtered)", f"{fraud_count_f:,}")
    with col4f:
        st.metric("Amount Lost to Fraud (USD) (Filtered)", f"${amount_lost_to_fraud_f:,.2f}")
    with col5f:
        st.metric("Average Loss per Fraud (USD) (Filtered)", f"${avg_loss_per_fraud_f:,.2f}")

# info filter aktif (opsional)
    if st.session_state.selected_device:
        st.info(f"Filter aktif: Device Type = {st.session_state.selected_device}")
    else:
        st.info("Filter aktif: All Devices")

# visualisasi dataset

    PLOTLY_TEMPLATE = "plotly_white"

# Palet pastel global (dipakai di semua chart)
    PASTEL_SEQ = [
    "#A3C4F3", "#BDB2FF", "#FFC6FF", "#CAFFBF", "#FFADAD",
    "#FFD6A5", "#FDFFB6", "#9BF6FF", "#Caffbf", "#D0F4DE"
    ]

# Warna konsisten untuk Fraud_Flag di scatter (0 vs 1)
    FRAUD_COLOR_MAP = {
    0: "#A3C4F3",  # non-fraud (pastel biru)
    1: "#FFADAD"   # fraud (pastel merah)
    }

# Heatmap pastel
    PASTEL_HEATMAP = ["#F8FAFC", "#E0E7FF", "#C7D2FE", "#A5B4FC", "#818CF8"]

    def apply_chart_style(fig, title_idn: str):
        fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=title_idn, x=0.0, xanchor="left"),
        font=dict(family="Arial", size=13, color="rgba(15,23,42,.90)"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(
            title="",
            bgcolor="rgba(255,255,255,.65)",
            bordercolor="rgba(15,23,42,.10)",
            borderwidth=1
        )
    )
        fig.update_xaxes(showgrid=True, gridcolor="rgba(15,23,42,.08)", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(15,23,42,.08)", zeroline=False)
        return fig


    st.dataframe(df_filtered.head(5))

    st.subheader("Distribusi Fraud Berdasarkan:")

    fraud_df = df_filtered[df_filtered["Fraud_Flag"] == 1]
    col1, col2, col3 = st.columns([5, 5, 5])

    with col1:
        pie_device = fraud_df.groupby("Device_Type").size().reset_index(name="counts")
        fig_device = px.pie(
        pie_device,
        names="Device_Type",
        values="counts",
        hole=0.4,
        color_discrete_sequence=PASTEL_SEQ
        )
        fig_device.update_traces(textposition="inside", textinfo="percent")
        fig_device = apply_chart_style(fig_device, "Jenis Perangkat")
        st.plotly_chart(fig_device, use_container_width=True)

    with col2:
        pie_card = fraud_df.groupby("Card_Type").size().reset_index(name="counts")
        fig_card = px.pie(
        pie_card,
        names="Card_Type",
        values="counts",
        hole=0.4,
        color_discrete_sequence=PASTEL_SEQ
        )
        fig_card.update_traces(textposition="inside", textinfo="percent")
        fig_card = apply_chart_style(fig_card, "Jenis Kartu")
        st.plotly_chart(fig_card, use_container_width=True)

    with col3:
        pie_trans = fraud_df.groupby("Transaction_Type").size().reset_index(name="counts")
        fig_trans = px.pie(
        pie_trans,
        names="Transaction_Type",
        values="counts",
        hole=0.4,
        color_discrete_sequence=PASTEL_SEQ
        )
        fig_trans.update_traces(textposition="inside", textinfo="percent")
        fig_trans = apply_chart_style(fig_trans, "Jenis Transaksi")
        st.plotly_chart(fig_trans, use_container_width=True)


# amount lost to fraud per merchant category (top 10)
    fraud_data = df_filtered[df_filtered['Fraud_Flag'] == 1]

    fraud_merchant = (
    fraud_data
    .groupby('Merchant_Category')['Amount']
    .sum()
    .reset_index()
    )

    top_10_merchant = fraud_merchant.nlargest(10, 'Amount')

    fig_merchant = px.bar(
    top_10_merchant,
    x='Merchant_Category',
    y='Amount',
    text='Amount',
    title='Top 10 Kategori Merchant dengan Kerugian Fraud Terbesar',
    labels={
        'Amount': 'Total Kerugian Fraud (USD)',
        'Merchant_Category': 'Kategori Merchant'
    },
    color_discrete_sequence=PASTEL_SEQ
    )

    fig_merchant.update_traces(
    texttemplate='$%{text:,.2f}',
    textposition='inside'
    )

    fig_merchant.update_layout(
    yaxis_tickprefix='$',
    yaxis_tickformat=','
    )
    fig_merchant = apply_chart_style(fig_merchant, "Top 10 Kategori Merchant dengan Kerugian Fraud Terbesar")
    st.plotly_chart(fig_merchant, use_container_width=True)


    df_filtered["Transaction_Date"] = pd.to_datetime(
    df_filtered["Transaction_Date"],
    errors="coerce"
    )
    df_filtered = df_filtered.dropna(subset=["Transaction_Date"])


    # line chart : fraud by day of week
    dow_map = {0:"Senin", 1:"Selasa", 2:"Rabu", 3:"Kamis", 4:"Jumat", 5:"Sabtu", 6:"Minggu"}
    dow_order = ["Senin","Selasa","Rabu","Kamis","Jumat","Sabtu","Minggu"]

    tmp = df_filtered.copy()
    tmp["day_of_week"] = tmp["Transaction_Date"].dt.dayofweek.map(dow_map)

    agg = (tmp.groupby("day_of_week", as_index=False)
       .agg(total_tx=("Transaction_ID","count"),
            fraud_tx=("Fraud_Flag","sum")))
    agg["fraud_rate"] = (agg["fraud_tx"] / agg["total_tx"]) * 100
    agg["day_of_week"] = pd.Categorical(agg["day_of_week"], categories=dow_order, ordered=True)
    agg = agg.sort_values("day_of_week")

    fig = px.line(
    agg,
    x="day_of_week",
    y="fraud_rate",
    markers=True,
    title="Persentase Fraud per Hari (Senin–Minggu)",
    color_discrete_sequence=PASTEL_SEQ
    )
    fig = apply_chart_style(fig, "Persentase Fraud per Hari (Senin–Minggu)")
    st.plotly_chart(fig, use_container_width=True)


# Donut: Fraud Distribution by Time of Day
    def time_of_day(hour: int) -> str:
        if 5 <= hour <= 10:
            return "Pagi"
        elif 11 <= hour <= 14:
            return "Siang"
        elif 15 <= hour <= 18:
            return "Sore"
        else:
            return "Malam"

    tmp = df_filtered.copy()
    tmp["time_of_day"] = tmp["Hour_of_Day"].astype(int).apply(time_of_day)
    agg = (tmp.groupby("time_of_day", as_index=False)
       .agg(total_tx=("Transaction_ID","count"),
            fraud_tx=("Fraud_Flag","sum")))

    order = ["Pagi","Siang","Sore","Malam"] 
    agg["time_of_day"] = pd.Categorical(agg["time_of_day"], categories=order, ordered=True)
    agg = agg.sort_values("time_of_day")

    fig = px.pie(
    agg,
    names="time_of_day",
    values="fraud_tx",
    hole=0.45,
    title="Distribusi Fraud berdasarkan Waktu (Pagi–Malam)",
    color_discrete_sequence=PASTEL_SEQ
    )
    fig.update_traces(textposition="inside", textinfo="percent")
    fig = apply_chart_style(fig, "Distribusi Fraud berdasarkan Waktu (Pagi–Malam)")
    st.plotly_chart(fig, use_container_width=True)


# Heatmap Fraud Rate: Device x Hour
    tmp = (df_filtered
       .groupby(["Device_Type", "Hour_of_Day"], as_index=False)
       .agg(
           total_tx=("Transaction_ID", "count"),
           fraud_tx=("Fraud_Flag", "sum")
       ))

    tmp["fraud_rate"] = (tmp["fraud_tx"] / tmp["total_tx"]) * 100

    pivot = (tmp.pivot(index="Device_Type", columns="Hour_of_Day", values="fraud_rate")
         .fillna(0)
         .sort_index(axis=0)
         .sort_index(axis=1))

    pivot_round = pivot.round(2)

    fig = px.imshow(
    pivot_round,
    aspect="auto",
    text_auto=True,
    title="Heatmap Persentase Fraud — Perangkat vs Jam",
    color_continuous_scale=PASTEL_HEATMAP
    )
    fig.update_layout(
    xaxis_title="Jam (Hour of Day)",
    yaxis_title="Jenis Perangkat"
    )
    fig = apply_chart_style(fig, "Heatmap Persentase Fraud — Perangkat vs Jam")
    st.plotly_chart(fig, use_container_width=True)


# Domestic vs International Fraud
    tmp = df_filtered.copy()

    tmp["Is_International_Label"] = tmp["Is_International"].map({
    0: "Domestic",
    1: "International"
    })

    agg = (
    tmp.groupby("Is_International_Label", as_index=False)
    .agg(
        total_tx=("Transaction_ID", "count"),
        fraud_tx=("Fraud_Flag", "sum"),
        fraud_loss=("Amount", lambda s: s[tmp.loc[s.index, "Fraud_Flag"] == 1].sum())
    )
    )

    agg["fraud_rate"] = (agg["fraud_tx"] / agg["total_tx"]) * 100
    agg["fraud_rate"] = agg["fraud_rate"].round(2)
    agg["fraud_loss"] = agg["fraud_loss"].round(2)

    order = ["Domestic", "International"]
    agg["Is_International_Label"] = pd.Categorical(
    agg["Is_International_Label"], categories=order, ordered=True
    )
    agg = agg.sort_values("Is_International_Label")

    col1, col2 = st.columns(2)

    with col1:
        fig_rate = px.bar(
        agg,
        y="Is_International_Label",
        x="fraud_rate",
        orientation="h",
        text="fraud_rate",
        title="Persentase Fraud — Domestik vs Internasional",
        color_discrete_sequence=PASTEL_SEQ
        )
        fig_rate = apply_chart_style(fig_rate, "Persentase Fraud — Domestik vs Internasional")
        st.plotly_chart(fig_rate, use_container_width=True)

    with col2:
        fig_loss = px.bar(
        agg,
        y="Is_International_Label",
        x="fraud_loss",
        orientation="h",
        text="fraud_loss",
        title="Kerugian Fraud (USD) — Domestik vs Internasional",
        color_discrete_sequence=PASTEL_SEQ
        )
    fig_loss = apply_chart_style(fig_loss, "Kerugian Fraud (USD) — Domestik vs Internasional")
    st.plotly_chart(fig_loss, use_container_width=True)


# fraud transaction : with chip&pin vs without
    show_metric = "Fraud Count"
    tmp = df_filtered.copy()
    tmp["chip_pin_group"] = ((tmp["Is_Chip"] == 1) & (tmp["Is_Pin_Used"] == 1)).map({
    True: "Chip&PIN Used",
    False: "Without Chip&PIN"
    })

    agg = (tmp.groupby("chip_pin_group", as_index=False)
       .agg(total_tx=("Transaction_ID","count"),
            fraud_tx=("Fraud_Flag","sum")))

    agg["fraud_rate"] = (agg["fraud_tx"] / agg["total_tx"]) * 100

    y_col = "fraud_tx" if show_metric == "Fraud Count" else "fraud_rate"

    fig = px.bar(
    agg,
    x="chip_pin_group",
    y=y_col,
    text=y_col,
    title=f"Fraud — Chip&PIN vs Tanpa Chip&PIN ({show_metric})",
    color_discrete_sequence=PASTEL_SEQ
    )
    fig = apply_chart_style(fig, f"Fraud — Chip&PIN vs Tanpa Chip&PIN ({show_metric})")
    st.plotly_chart(fig, use_container_width=True)


    # Scatter Plot: Distance vs Amount
    sample = df_filtered.sample(
    n=min(len(df_filtered), 20000),
    random_state=42
)

    fig_scatter = px.scatter(
    sample,
    x="Distance_From_Home",
    y="Amount",
    color="Fraud_Flag",
    opacity=0.6,
    title="Scatter: Jarak dari Rumah vs Nilai Transaksi (Sampel)",
    color_discrete_map=FRAUD_COLOR_MAP
    )
    fig = apply_chart_style(fig_scatter, "Jarak dari Rumah vs Nilai Transaksi (Sampel)")
    st.plotly_chart(fig, use_container_width=True)


    # Line Plot: Fraud Rate by Distance Range
    tmp = df_filtered.copy()

    bins = [-0.0001, 1, 5, 10, 25, 50, 100, float("inf")]
    labels = ["<=1km", "1–5km", "5–10km", "10–25km", "25–50km", "50–100km", ">100km"]

    tmp["distance_range"] = pd.cut(
    tmp["Distance_From_Home"],
    bins=bins,
    labels=labels
    )   

    agg = (
    tmp.groupby("distance_range", as_index=False)
    .agg(
        total_tx=("Transaction_ID", "count"),
        fraud_tx=("Fraud_Flag", "sum")
    )
    )

    agg["fraud_rate"] = (agg["fraud_tx"] / agg["total_tx"]) * 100

    fig_line = px.line(
    agg,
    x="distance_range",
    y="fraud_rate",
    markers=True,
    title="Persentase Fraud berdasarkan Rentang Jarak dari Rumah",
    color_discrete_sequence=PASTEL_SEQ
    )
    fig_line = apply_chart_style(fig_line, "Persentase Fraud berdasarkan Rentang Jarak dari Rumah")
    st.plotly_chart(fig_line, use_container_width=True)


    # fraud by high deviation (Normal vs Anomali)
    show_metric = "Fraud Count"
    tmp = df_filtered.copy()
    mean_amt = tmp["Amount"].mean()
    std_amt = tmp["Amount"].std(ddof=0)
    std_amt = std_amt if std_amt != 0 else 1

    tmp["z_amount"] = (tmp["Amount"] - mean_amt) / std_amt
    tmp["behavior_group"] = tmp["z_amount"].abs().apply(
    lambda z: "Anomali" if z > 3 else "Normal Behavior"
    )

    agg = (tmp.groupby("behavior_group", as_index=False)
       .agg(total_tx=("Transaction_ID","count"),
            fraud_tx=("Fraud_Flag","sum"),
            fraud_loss=("Amount", lambda s: s[tmp.loc[s.index, "Fraud_Flag"] == 1].sum())))

    agg["fraud_rate"] = (agg["fraud_tx"] / agg["total_tx"]) * 100

    order = ["Normal Behavior", "Anomali"]
    agg["behavior_group"] = pd.Categorical(agg["behavior_group"], categories=order, ordered=True)
    agg = agg.sort_values("behavior_group")

    y_col = "fraud_tx" if show_metric == "Fraud Count" else "fraud_rate"

    c1, c2 = st.columns(2)

    with c1:
        fig = px.bar(
        agg,
        y="behavior_group",
        x=y_col,
        orientation="h",
        text=y_col,
        title=f"Fraud — Perilaku Normal vs Anomali ({show_metric})",
        color_discrete_sequence=PASTEL_SEQ
    )
        fig = apply_chart_style(fig, f"Fraud — Perilaku Normal vs Anomali ({show_metric})")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.bar(
        agg,
        y="behavior_group",
        x="fraud_loss",
        orientation="h",
        text="fraud_loss",
        title="Kerugian Fraud (USD) — Normal vs Anomali",
        color_discrete_sequence=PASTEL_SEQ
        )
        fig2 = apply_chart_style(fig2, "Kerugian Fraud (USD) — Normal vs Anomali")
        st.plotly_chart(fig2, use_container_width=True)


# Top 10 Country by Fraud Count
    fraud_country = (
    df_filtered[df_filtered["Fraud_Flag"] == 1]
    .groupby("Country")
    .size()
    .reset_index(name="fraud_count")
    .sort_values("fraud_count", ascending=False)
    .head(10)
    )

    fig_country = px.bar(
    fraud_country,
    x="Country",
    y="fraud_count",
    text="fraud_count",
    title="Top 10 Negara dengan Transaksi Fraud Terbanyak",
    color_discrete_sequence=PASTEL_SEQ
    )

    fig_country.update_traces(textposition="inside")

    fig_country.update_layout(
    xaxis_title="Negara",
    yaxis_title="Jumlah Transaksi Fraud"
    )

    fig_country = apply_chart_style(fig_country, "Top 10 Negara dengan Transaksi Fraud Terbanyak")
    st.plotly_chart(fig_country, use_container_width=True)


