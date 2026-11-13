# bank-customer-churn
Customer Churn Analysis & Machine Learning Prediction System in the Banking Industry: Identifying Key Drivers and Retaining High-Risk Customer Segments.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bankcustomerchurn2323.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6.1-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-green.svg)](https://xgboost.readthedocs.io/)

---

## 🌐 Live Interactive Web Application (Streamlit)

Aplikasi web interaktif telah dibangun dan dapat diakses publik oleh siapa saja secara gratis melalui **Streamlit Community Cloud**:

👉 **[Buka Aplikasi Streamlit Live (bankcustomerchurn2323.streamlit.app)](https://bankcustomerchurn2323.streamlit.app)**  

---

## 📌 Project Overview

Customer churn adalah salah satu tantangan terbesar dalam industri perbankan. Kehilangan nasabah tidak hanya mengurangi potensi pendapatan dan likuiditas dana pihak ketiga (DPK), tetapi juga meningkatkan biaya akuisisi (*customer acquisition cost*). Proyek ini bertujuan menganalisis perilaku churn nasabah Bank XYZ, mengidentifikasi faktor pemicu utama, serta membangun model Machine Learning terbaik yang terintegrasi ke dalam aplikasi web Streamlit sebagai **Sistem Peringatan Dini (Early Warning System)** bagi tim CRM dan Marketing.

---

## 🎯 Business Problem & Questions

Bank XYZ ingin menekan angka customer churn dengan menjawab pertanyaan bisnis utama:

1. **Demografi:** Kelompok nasabah mana yang memiliki risiko churn paling tinggi berdasarkan jenis kelamin, usia, dan negara?
2. **Produk & Layanan:** Seberapa besar pengaruh jumlah produk serta kepemilikan kartu kredit terhadap retensi nasabah?
3. **Masa Hubungan (Tenure):** Pola churn apa yang muncul seiring bertambahnya lama nasabah menjadi anggota?
4. **Keluhan Nasabah (Complain):** Apakah pengajuan komplain menjadi sinyal mutlak sebelum nasabah menutup rekening?
5. **Dampak Finansial:** Bagaimana perbandingan saldo nasabah yang keluar terhadap keuntungan dan likuiditas bank?
6. **Keaktifan (IsActiveMember):** Apakah status keaktifan nasabah terbukti signifikan secara statistik memengaruhi churn?

---

## 📂 Dataset Information

Sumber: [Kaggle Bank Customer Churn Dataset](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)

Dataset memuat **10.000 data historis nasabah** dengan variabel:
- `CustomerId`: ID unik nasabah (dihapus saat modeling)
- `Surname`: Nama keluarga nasabah (dihapus saat modeling)
- `CreditScore`: Skor kelayakan kredit nasabah (350 - 850)
- `Geography`: Negara domisili (France, Germany, Spain)
- `Gender`: Jenis kelamin (Female, Male)
- `Age`: Usia nasabah
- `Tenure`: Lama menjadi nasabah (0 - 10 tahun)
- `Balance`: Saldo rekening simpanan (€)
- `NumOfProducts`: Jumlah produk bank yang dimiliki (1 - 4)
- `HasCrCard`: Kepemilikan kartu kredit (1 = Ya, 0 = Tidak)
- `IsActiveMember`: Status keaktifan transaksi nasabah (1 = Aktif, 0 = Inaktif)
- `EstimatedSalary`: Estimasi pendapatan tahunan (€)
- `Satisfaction Score`: Skor kepuasan pelanggan (1 - 5)
- `Card Type`: Jenis kartu kredit (DIAMOND, GOLD, PLATINUM, SILVER)
- `Point Earned`: Poin reward loyalitas nasabah
- `Complain`: Riwayat pernah komplain (1 = Ya, 0 = Tidak)
- `Exited`: **Target Variable** (1 = Churn / Keluar, 0 = Stay / Bertahan)

---

## 🔍 Ringkasan Temuan EDA (Exploratory Data Analysis)

Dari eksplorasi terhadap 10.000 data nasabah (2.038 Churn, 7.962 Loyal):

1. **Gender:** Nasabah perempuan memiliki churn rate jauh lebih tinggi (**25.1%**) dibandingkan nasabah laki-laki (**16.5%**).
2. **Usia:** Churn melonjak drastis pada nasabah senior usia **41–50 tahun (34.0%)** dan **51+ tahun (44.6%)**, dibandingkan usia 18–30 (7.5%).
3. **Geografi:** Nasabah asal **Jerman memiliki churn rate 32.54%** (hampir 2x lipat dibanding Perancis 16.2% dan Spanyol 16.7%).
4. **Multi-Product Paradox:** Nasabah dengan **2 produk paling loyal (churn hanya 7.6%)**. Namun, nasabah dengan **3 produk (82.7%)** dan **4 produk (100%)** mengalami churn ekstrem akibat beban biaya administrasi dan kompleksitas layanan.
5. **Indikator Komplain:** **99.5% nasabah yang churn pernah mengajukan komplain**. Komplain adalah alarm penutupan rekening yang paling krusial.
6. **Dampak Saldo (Financial Drain):** Rata-rata saldo nasabah churn adalah **€91.109**, jauh lebih tinggi dibanding nasabah bertahan (**€72.743**). Total potensi dana keluar mencapai lebih dari **€185 Juta**.
7. **Signifikansi Statistik Keaktifan:** Uji Chi-Square membuktikan hubungan signifikan ($p < 0.001$). Nasabah inaktif pada usia 51+ tahun memiliki churn rate tertinggi mencapai **84.94%**.

---

## 🤖 Machine Learning Modeling

### 1. Data Splitting & Feature Selection
- Fitur identifier non-prediktif (`CustomerId`, `Surname`, `RowNumber`) dihapus.
- Kolom `Complain` dihapus dari data modeling karena memiliki korelasi 99.5% dengan target (mencegah *data leakage* fatal).
- Dataset dibagi menjadi **80% Training (8.000)** dan **20% Testing (2.000)** dengan `stratify=y` untuk menjaga proporsi kelas churn (20.38%).

### 2. Preprocessing Pipeline (`ColumnTransformer`)
Mencegah *data leakage* dengan membungkus transformasi ke dalam satu pipeline:
- **Numerical Features** (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `EstimatedSalary`, `Satisfaction Score`, `Point Earned`):
  - `SimpleImputer(strategy='median')`
  - `StandardScaler()`
- **Categorical Features** (`Geography`, `Gender`, `Card Type`):
  - `SimpleImputer(strategy='most_frequent')`
  - `OneHotEncoder(drop='first', handle_unknown='ignore')`
- **Binary Features** (`HasCrCard`, `IsActiveMember`):
  - `SimpleImputer(strategy='most_frequent')`

### 3. Penanganan Data Imbalance
Rasio data adalah 80:20 (Imbalanced). Model diberikan bobot penyeimbang:
$$\text{scale\_pos\_weight} = \frac{\text{Negative Class (Stay)}}{\text{Positive Class (Churn)}} \approx 3.91$$

### 4. Algoritma & Hyperparameter Tuning
Eksperimen dilakukan terhadap 5 algoritma (Logistic Regression, SVC, Decision Tree, Random Forest, dan XGBoost). Model dengan performa pemisahan terbaik adalah **XGBoost Classifier** yang kemudian di-tuning menggunakan `GridSearchCV` (5-Fold Stratified CV, scoring: `f1`):
* `learning_rate`: `0.1`
* `max_depth`: `3`
* `n_estimators`: `150`
* `subsample`: `0.8`

### 5. Hasil Evaluasi Performa Model (Test Set: 2.000 Nasabah)

| Metrik Evaluasi | Nilai | Interpretasi Bisnis |
| :--- | :---: | :--- |
| **ROC-AUC Score** | **0.8771** | Kemampuan diskriminasi model sangat kuat dalam memisahkan nasabah berisiko dan loyal |
| **Recall (Class 1 Churn)** | **76.0%** | Mampu menangkap 76 dari setiap 100 nasabah yang berpotensi churn (minim *False Negative*) |
| **Akurasi** | **79.95% (~80%)** | Tingkat ketepatan prediksi keseluruhan data uji |
| **F1-Score** | **0.6072** | Keseimbangan harmonis antara Precision dan Recall pada kelas minoritas |
| **Precision (Class 1)** | **50.57%** | Lebih dari 50% nasabah yang ditandai berisiko terbukti benar-benar akan keluar |

> **Mengapa Recall & ROC-AUC Diprioritaskan?**  
> Dalam strategi retensi perbankan, biaya kehilangan seorang nasabah bernilai tinggi (*False Negative*) jauh lebih besar daripada biaya mengirimkan penawaran promo loyalitas kepada nasabah yang sebenarnya bertahan (*False Positive*).

### 6. Feature Importance (Faktor Paling Berpengaruh)
1. **NumOfProducts (19.5%):** Faktor paling dominan dalam memprediksi churn.
2. **Age (16.5%):** Usia nasabah (terutama kelompok 41+).
3. **IsActiveMember (14.2%):** Keaktifan transaksi nasabah.
4. **Geography_Germany (9.8%):** Pengaruh pasar nasabah Jerman.
5. **Gender_Male (7.5%):** Variasi risiko berdasarkan gender.
6. **Balance (5.6%):** Tingkat saldo simpanan nasabah.

---

## 💻 Fitur Aplikasi Streamlit (`app.py`)

1. **📊 Hasil Analisis EDA:** Eksplorasi interaktif 6 pertanyaan riset dengan grafik Plotly, metrik KPI, dan uji hipotesis Chi-Square.
2. **🎯 Prediksi Individu (Simulator):** Form input profil nasabah dengan output probabilitas churn, tingkat risiko, dan **Targeted Retention Plan** otomatis.
3. **📁 Prediksi Massal (Batch CSV):** Upload file CSV nasabah, inferensi batch, visualisasi distribusi risiko, dan fitur unduh CSV hasil analisis.
4. **📈 Feature Importance & Strategi Bisnis:** Grafik signifikansi fitur XGBoost dan tabel matriks aksi retensi.
5. **ℹ️ Kamus Data & Info Model:** Dokumentasi pipeline dan arsitektur model.

### Cara Menjalankan Secara Lokal:
```bash
# 1. Clone repository
git clone https://github.com/RafTech19/bank-customer-churn.git
cd bank-customer-churn

# 2. Install dependensi
pip install -r requirements.txt

# 3. Jalankan Streamlit
streamlit run app.py
```

---

## 📊 Dashboard & Presentasi Lainnya

* **Tableau Public Interactive Dashboard:** [Lihat Dashboard Tableau](https://public.tableau.com/app/profile/muhammad.rafli.febriyanto/viz/P1M2_Muhamad_Rafli_Febriyanto/Dashboard1)
* **Canva Slide Presentation:** [Lihat Presentasi Proyek](https://www.canva.com/design/DAHDFUTXE5U/b60Ix1glrk4pL5ZAStczKw/edit?utm_content=DAHDFUTXE5U&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## 🛠️ Tools & Libraries

- **Programming:** Python 3.10+
- **Machine Learning & Pipeline:** Scikit-Learn, XGBoost, Joblib
- **Data Wrangling & Stats:** Pandas, NumPy, SciPy
- **Data Visualization & Web App:** Streamlit, Plotly, Seaborn, Matplotlib
- **BI Dashboard:** Tableau Public

---

## 👤 Author

**Muhammad Rafli Febriyanto**  
*Data Analyst & Machine Learning Enthusiast*  
Email: [rafli.bim05@gmail.com](mailto:rafli.bim05@gmail.com)  
GitHub: [@RafTech19](https://github.com/RafTech19)
