import streamlit as st

def about_dataset():
    st.write('**Tentang Dataset**')
    col1, col2= st.columns([5,5])

    with col1:
        link = "https://www.xenonstack.com/hs-fs/hubfs/xenonstack-credit-card-fraud-detection.png?width=1920&height=1080&name=xenonstack-credit-card-fraud-detection.png"
        st.image(link, caption="Credit Card Fraud Detection Dataset")

    with col2:
        st.write('**Fraud kartu kredit** adalah tindakan kejahatan di mana informasi kartu kredit ' \
        'orang lain dicuri atau disalahgunakan tanpa izin untuk melakukan transaksi ilegal, ' \
        'seperti pembelian barang, layanan, atau penarikan dana, tanpa sepengetahuan pemilik kartu.')
        st.markdown("""
    **Fraud kartu kredit** merupakan salah satu permasalahan kritis di industri keuangan digital, 
karena transaksi fraud dapat menyebabkan kerugian finansial yang besar serta menurunkan kepercayaan 
nasabah. Seiring meningkatnya transaksi non-tunai melalui mobile, web, dan terminal, pola fraud 
menjadi semakin kompleks dan sulit dideteksi secara manual. Oleh karena itu, pendekatan berbasis 
data dan machine learning menjadi solusi yang relevan untuk mengidentifikasi transaksi mencurigakan 
secara cepat dan konsisten.

Dataset yang digunakan dalam proyek ini merepresentasikan transaksi kartu kredit dengan berbagai 
karakteristik, seperti nilai transaksi, jarak lokasi, jenis perangkat, kategori merchant, serta 
fitur keamanan (Chip & PIN). Tujuan utama analisis adalah membangun model klasifikasi yang mampu 
mendeteksi transaksi fraud secara akurat, dengan fokus utama pada meminimalkan **False Negative** 
(fraud yang lolos), karena kesalahan ini memiliki dampak bisnis yang paling berisiko.
""")

