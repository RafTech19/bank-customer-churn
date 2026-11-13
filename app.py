import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency

# ---------------------------------------------------------
# Page Configuration & Advanced SaaS CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bank Churn AI • Retention Analytics & Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional SaaS Theme styling inspired by modern design systems
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #F8FAFC;
    }

    /* Modern top navigation bar */
    .saas-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #FFFFFF;
        padding: 14px 24px;
        border-radius: 14px;
        border: 1px solid #E2E8F0;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .saas-header-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .saas-brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .saas-brand-tag {
        background: #EEF2FF;
        color: #4F46E5;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid #C7D2FE;
    }
    .saas-user-badge {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #F1F5F9;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #334155;
    }
    .saas-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: #4F46E5;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 800;
    }

    /* Modern card containers */
    .saas-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    }
    .saas-card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .saas-card-desc {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 16px;
    }

    /* Metric cards */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: #94A3B8;
    }

    /* Status badges */
    .badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-success {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }

    /* Callout & result box */
    .callout-box {
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        border-left: 5px solid;
    }
    .callout-high {
        background: #FFF1F2;
        border-color: #E11D48;
        color: #881337;
    }
    .callout-medium {
        background: #FFFBEB;
        border-color: #F59E0B;
        color: #78350F;
    }
    .callout-low {
        background: #F0FDF4;
        border-color: #10B981;
        color: #064E3B;
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 16px;
        border-bottom: 1px solid #E2E8F0;
    }
    .sidebar-brand h2 {
        font-size: 1.15rem;
        font-weight: 800;
        color: #0F172A;
        margin: 0;
    }
    .sidebar-brand p {
        font-size: 0.78rem;
        color: #64748B;
        margin: 0;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #F1F5F9;
        padding: 4px;
        border-radius: 10px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.85rem;
        color: #475569;
        background-color: transparent;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4F46E5 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Data & Model Pipeline with Caching
# ---------------------------------------------------------
DATA_PATH = "customer-churn.csv"
MODEL_PATH = "best_xgboost_pipeline.pkl"

@st.cache_data
def load_dataset():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    # Define AgeGroup if not present for consistent EDA
    bins = [18, 30, 40, 50, 100]
    labels = ['18-30', '31-40', '41-50', '51+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels)
    return df

@st.cache_resource
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

df_raw = load_dataset()
pipeline = load_pipeline()

# ---------------------------------------------------------
# Top Header Bar (SaaS Style)
# ---------------------------------------------------------
st.markdown("""
<div class="saas-header">
    <div class="saas-header-left">
        <span style="font-size: 1.8rem;">🏦</span>
        <div>
            <div style="display:flex; align-items:center; gap:8px;">
                <h1 class="saas-brand-title">Bank Churn Analytics & AI</h1>
                <span class="saas-brand-tag">v2.0 Tuned</span>
            </div>
            <div style="font-size: 0.82rem; color: #64748B;">Platform Intelijen Retensi & Deteksi Dini Churn Nasabah Bank XYZ</div>
        </div>
    </div>
    <div class="saas-user-badge">
        <div class="saas-avatar">MR</div>
        <div>
            <div style="font-size: 0.82rem; font-weight: 700; color: #0F172A;">M. Rafli Febriyanto</div>
            <div style="font-size: 0.72rem; color: #64748B;">Lead Data Analyst</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Navigation Menu
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="background:#4F46E5; width:36px; height:36px; border-radius:10px; display:flex; align-items:center; justify-content:center; color:white; font-size:1.1rem; font-weight:bold;">
            BC
        </div>
        <div>
            <h2>Bank Churn AI</h2>
            <p>Intelligence Workspace</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio(
        "PILIHAN NAVIGASI",
        [
            "📊 Hasil Analisis EDA",
            "🎯 Prediksi Individu (Simulator)",
            "📁 Prediksi Massal (Batch CSV)",
            "📈 Feature Importance & Strategi",
            "ℹ️ Kamus Data & Info Model"
        ]
    )

    st.markdown("---")
    st.markdown("### ⚡ Ringkasan Status")
    st.markdown("""
    * **Model:** Tuned XGBoost
    * **ROC-AUC:** `0.8771`
    * **Recall Churn:** `76.0%`
    * **Dataset:** 10,000 Nasabah
    * **Tingkat Churn:** `20.38%`
    """)
    st.caption("Proyek Portofolio Data Analyst • 2026")

# ---------------------------------------------------------
# MENU 1: HASIL ANALISIS EDA (DARI NOTEBOOK)
# ---------------------------------------------------------
if menu == "📊 Hasil Analisis EDA":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size:1.6rem; font-weight:800; color:#0F172A; margin-bottom:4px;">📊 Exploratory Data Analysis (EDA)</h2>
        <div style="color:#64748B; font-size:0.92rem;">Hasil eksplorasi mendalam dari 10.000 data historis nasabah Bank XYZ untuk mengungkap faktor pemicu utama customer churn.</div>
    </div>
    """, unsafe_allow_html=True)

    if df_raw is None:
        st.error("Dataset `customer-churn.csv` tidak ditemukan.")
        st.stop()

    # KPI Summary Cards at the top
    total_cust = len(df_raw)
    churn_cust = (df_raw['Exited'] == 1).sum()
    stay_cust = (df_raw['Exited'] == 0).sum()
    churn_pct = (churn_cust / total_cust) * 100
    avg_balance_churn = df_raw[df_raw['Exited'] == 1]['Balance'].mean()
    total_balance_at_risk = df_raw[df_raw['Exited'] == 1]['Balance'].sum()

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-label">Total Nasabah</div>
            <div class="kpi-value">{total_cust:,}</div>
            <div class="kpi-sub">Data historis bank</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Nasabah Churn</div>
            <div class="kpi-value" style="color: #DC2626;">{churn_cust:,}</div>
            <div class="kpi-sub"><span class="badge-danger">{churn_pct:.2f}% Total Rasio</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Nasabah Loyal (Stay)</div>
            <div class="kpi-value" style="color: #16A34A;">{stay_cust:,}</div>
            <div class="kpi-sub"><span class="badge-success">{100-churn_pct:.2f}% Bertahan</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Saldo Nasabah Churn</div>
            <div class="kpi-value" style="color: #4F46E5;">€{avg_balance_churn:,.0f}</div>
            <div class="kpi-sub">Rata-rata saldo keluar</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs for 6 Research Questions from Notebook
    eda_tabs = st.tabs([
        "4.1 Kondisi Demografis",
        "4.2 Produk & Layanan",
        "4.3 Tren Tenure",
        "4.4 Komplain & Kepuasan",
        "4.5 Dampak Finansial",
        "4.6 Uji Keaktifan (Chi-Square)"
    ])

    # ---------------- 4.1 DEMOGRAFI ----------------
    with eda_tabs[0]:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">📌 4.1 Risiko Churn Berdasarkan Kondisi Demografis</div>
            <div class="saas-card-desc">Menganalisis sebaran risiko churn berdasarkan jenis kelamin, kelompok usia, dan wilayah geografi nasabah.</div>
        """, unsafe_allow_html=True)

        col_d1, col_d2, col_d3 = st.columns(3)

        with col_d1:
            # Gender breakdown
            df_gender = df_raw.groupby('Gender')['Exited'].agg(Total='count', Churn='sum').reset_index()
            df_gender['Churn_Rate'] = (df_gender['Churn'] / df_gender['Total']) * 100
            
            fig_gender = px.bar(
                df_gender,
                x='Gender',
                y='Churn_Rate',
                color='Gender',
                text=df_gender['Churn_Rate'].apply(lambda x: f"{x:.1f}%"),
                title='Churn Rate Berdasarkan Gender',
                color_discrete_map={'Female': '#EC4899', 'Male': '#3B82F6'}
            )
            fig_gender.update_layout(yaxis=dict(title='Churn Rate (%)', range=[0, 35]), showlegend=False, height=320)
            st.plotly_chart(fig_gender, use_container_width=True)

        with col_d2:
            # Age Group breakdown
            df_age = df_raw.groupby('AgeGroup')['Exited'].agg(Total='count', Churn='sum').reset_index()
            df_age['Churn_Rate'] = (df_age['Churn'] / df_age['Total']) * 100

            fig_age = px.bar(
                df_age,
                x='AgeGroup',
                y='Churn_Rate',
                text=df_age['Churn_Rate'].apply(lambda x: f"{x:.1f}%"),
                title='Churn Rate Berdasarkan Kelompok Usia',
                color='Churn_Rate',
                color_continuous_scale='Reds'
            )
            fig_age.update_layout(yaxis=dict(title='Churn Rate (%)', range=[0, 55]), showlegend=False, height=320)
            st.plotly_chart(fig_age, use_container_width=True)

        with col_d3:
            # Geography breakdown
            df_geo = df_raw.groupby('Geography')['Exited'].agg(Total='count', Churn='sum').reset_index()
            df_geo['Churn_Rate'] = (df_geo['Churn'] / df_geo['Total']) * 100

            fig_geo = px.bar(
                df_geo,
                x='Geography',
                y='Churn_Rate',
                text=df_geo['Churn_Rate'].apply(lambda x: f"{x:.1f}%"),
                title='Churn Rate Berdasarkan Negara',
                color='Geography',
                color_discrete_map={'France': '#60A5FA', 'Germany': '#EF4444', 'Spain': '#F59E0B'}
            )
            fig_geo.update_layout(yaxis=dict(title='Churn Rate (%)', range=[0, 40]), showlegend=False, height=320)
            st.plotly_chart(fig_geo, use_container_width=True)

        st.markdown("""
        <div style="background:#F8FAFC; border-radius:10px; padding:14px; border:1px solid #E2E8F0; margin-top:8px;">
            <div style="font-weight:700; color:#0F172A; margin-bottom:4px;">💡 Insight & Rekomendasi Demografis:</div>
            <ul style="margin:0; padding-left:20px; font-size:0.88rem; color:#334155;">
                <li><b>Gender:</b> Nasabah perempuan memiliki churn rate jauh lebih tinggi (<b>25.07%</b> vs pria <b>16.47%</b>). <i>Rekomendasi:</i> Kampanye loyalitas lifestyle & cashback belanja.</li>
                <li><b>Usia:</b> Churn melonjak drastis pada kelompok usia <b>41-50 tahun (34.0%)</b> dan <b>51+ tahun (44.6%)</b>. <i>Rekomendasi:</i> Produk wealth management, dana pensiun, serta pendampingan digital banking di kantor cabang.</li>
                <li><b>Geografi:</b> Nasabah <b>Jerman (32.54%)</b> memiliki risiko churn 2x lipat dibanding Perancis (16.2%) dan Spanyol (16.7%). <i>Rekomendasi:</i> Benchmarking suku bunga simpanan dan biaya admin terhadap kompetitor perbankan di Jerman.</li>
            </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- 4.2 PRODUK & LAYANAN ----------------
    with eda_tabs[1]:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">📌 4.2 Pengaruh Jumlah Produk & Kartu Kredit terhadap Churn</div>
            <div class="saas-card-desc">Membuktikan hipotesis apakah memiliki banyak produk membuat nasabah lebih loyal atau justru sebaliknya.</div>
        """, unsafe_allow_html=True)

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            # Num of products breakdown
            df_prod = df_raw.groupby('NumOfProducts')['Exited'].agg(Total='count', Churn='sum').reset_index()
            df_prod['Churn_Rate'] = (df_prod['Churn'] / df_prod['Total']) * 100
            
            fig_prod = px.bar(
                df_prod,
                x=df_prod['NumOfProducts'].astype(str),
                y='Churn_Rate',
                text=df_prod['Churn_Rate'].apply(lambda x: f"{x:.1f}%"),
                title='Churn Rate Berdasarkan Jumlah Produk (NumOfProducts)',
                color='Churn_Rate',
                color_continuous_scale='Reds'
            )
            fig_prod.update_layout(xaxis=dict(title='Jumlah Produk Dimiliki'), yaxis=dict(title='Churn Rate (%)', range=[0, 110]), height=340)
            st.plotly_chart(fig_prod, use_container_width=True)

        with col_p2:
            # Credit card breakdown
            df_cc = df_raw[df_raw['Exited'] == 1]['HasCrCard'].value_counts().reset_index()
            df_cc.columns = ['HasCrCard', 'Count']
            df_cc['Label'] = df_cc['HasCrCard'].map({1: 'Punya Kartu Kredit', 0: 'Tidak Punya Kartu Kredit'})

            fig_cc = px.pie(
                df_cc,
                names='Label',
                values='Count',
                title='Komposisi Kepemilikan Kartu Kredit pada Nasabah Churn',
                color='Label',
                color_discrete_map={'Punya Kartu Kredit': '#EF4444', 'Tidak Punya Kartu Kredit': '#94A3B8'},
                hole=0.45
            )
            fig_cc.update_layout(height=340)
            st.plotly_chart(fig_cc, use_container_width=True)

        st.markdown("""
        <div style="background:#FEF2F2; border-radius:10px; padding:14px; border:1px solid #FCA5A5;">
            <div style="font-weight:700; color:#991B1B; margin-bottom:4px;">🚨 Temuan Kritis (Multi-Product Paradox):</div>
            <div style="font-size:0.88rem; color:#7F1D1D;">
                • <b>Nasabah dengan 2 Produk adalah yang paling loyal</b> (churn rate hanya <b>7.6%</b>).<br>
                • Namun, saat nasabah memiliki <b>3 produk (82.7%)</b> atau <b>4 produk (100%)</b>, tingkat churn melonjak ekstrem! Hampir seluruh nasabah kelompok ini menutup rekening karena merasa terbebani oleh biaya administrasi terpisah atau ketidaknyamanan pengelolaan banyak akun.<br>
                • Selain itu, sebesar <b>69.9%</b> nasabah yang churn ternyata memiliki kartu kredit, menepis asumsi bahwa kartu kredit menjamin retensi.
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- 4.3 TREN TENURE ----------------
    with eda_tabs[2]:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">📌 4.3 Tren Churn Rate Berdasarkan Masa Hubungan (Tenure)</div>
            <div class="saas-card-desc">Menganalisis apakah lama waktu menjadi nasabah bank berpengaruh terhadap loyalitas nasabah.</div>
        """, unsafe_allow_html=True)

        df_tenure = df_raw.groupby('Tenure')['Exited'].agg(Total='count', Churn='sum').reset_index()
        df_tenure['Churn_Rate'] = (df_tenure['Churn'] / df_tenure['Total']) * 100

        fig_tenure = px.line(
            df_tenure,
            x='Tenure',
            y='Churn_Rate',
            markers=True,
            title='Tren Churn Rate Sepanjang Masa Hubungan Nasabah (0 - 10 Tahun)',
            labels={'Tenure': 'Tenure (Tahun Menjadi Nasabah)', 'Churn_Rate': 'Churn Rate (%)'}
        )
        fig_tenure.update_traces(line=dict(color='#4F46E5', width=3), marker=dict(size=8, color='#4338CA'))
        fig_tenure.update_layout(yaxis=dict(range=[15, 26]), height=360)
        st.plotly_chart(fig_tenure, use_container_width=True)

        st.markdown("""
        <div style="background:#F8FAFC; border-radius:10px; padding:14px; border:1px solid #E2E8F0;">
            <div style="font-weight:700; color:#0F172A; margin-bottom:4px;">💡 Insight Masa Hubungan (Tenure):</div>
            <div style="font-size:0.88rem; color:#334155;">
                Grafik menunjukkan pola kurva U-shape tipis: churn sedikit lebih tinggi pada <b>nasabah baru (tahun 0-1)</b> dan <b>nasabah lama (tahun 7-10)</b>. 
                Nasabah baru rentan keluar karena belum terbentuk kebiasaan finansial, sedangkan nasabah lama rentan tergiur oleh tawaran produk baru dari kompetitor.
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- 4.4 KOMPLAIN & KEPUASAN ----------------
    with eda_tabs[3]:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">📌 4.4 Hubungan Komplain & Skor Kepuasan terhadap Churn</div>
            <div class="saas-card-desc">Menilai apakah keluhan yang diajukan nasabah menjadi prediktor mutlak bagi customer churn.</div>
        """, unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            df_comp = df_raw.groupby('Complain')['Exited'].agg(Total='count', Churn='sum').reset_index()
            df_comp['Churn_Rate'] = (df_comp['Churn'] / df_comp['Total']) * 100
            df_comp['Status'] = df_comp['Complain'].map({1: 'Pernah Komplain', 0: 'Tidak Pernah Komplain'})

            fig_comp = px.bar(
                df_comp,
                x='Status',
                y='Churn_Rate',
                text=df_comp['Churn_Rate'].apply(lambda x: f"{x:.1f}%"),
                title='Churn Rate Berdasarkan Riwayat Komplain',
                color='Status',
                color_discrete_map={'Pernah Komplain': '#EF4444', 'Tidak Pernah Komplain': '#10B981'}
            )
            fig_comp.update_layout(yaxis=dict(title='Churn Rate (%)', range=[0, 110]), showlegend=False, height=340)
            st.plotly_chart(fig_comp, use_container_width=True)

        with col_c2:
            df_churn_comp = df_raw[(df_raw['Exited'] == 1) & (df_raw['Complain'] == 1)]
            fig_sat = px.histogram(
                df_churn_comp,
                x='Satisfaction Score',
                title='Distribusi Skor Kepuasan pada Nasabah Churn yang Komplain',
                color_discrete_sequence=['#F59E0B']
            )
            fig_sat.update_layout(xaxis=dict(title='Skor Kepuasan (1 - 5)'), yaxis=dict(title='Jumlah Nasabah'), height=340)
            st.plotly_chart(fig_sat, use_container_width=True)

        st.markdown("""
        <div style="background:#FEF2F2; border-radius:10px; padding:14px; border:1px solid #FCA5A5;">
            <div style="font-weight:700; color:#991B1B; margin-bottom:4px;">🚨 Indikator Churn Paling Mematikan:</div>
            <div style="font-size:0.88rem; color:#7F1D1D;">
                • Sebanyak <b>99.5% nasabah yang pernah mengajukan komplain berujung menutup rekeningnya</b>.<br>
                • <b>Paradoks Kepuasan:</b> Banyak nasabah yang memberikan skor kepuasan 4 atau 5 tetap melakukan komplain dan keluar, membuktikan bahwa komplain tiket adalah alarm retensi yang jauh lebih nyata daripada survey berkala.
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- 4.5 DAMPAK FINANSIAL ----------------
    with eda_tabs[4]:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">📌 4.5 Pengaruh Customer Churn terhadap Likuiditas Saldo Bank</div>
            <div class="saas-card-desc">Membandingkan profil saldo nasabah yang keluar versus nasabah yang bertahan.</div>
        """, unsafe_allow_html=True)

        df_bal = df_raw.groupby('Exited')['Balance'].agg(['mean', 'median', 'sum']).reset_index()
        df_bal['Status'] = df_bal['Exited'].map({1: 'Churn (Keluar)', 0: 'Stay (Bertahan)'})

        col_b1, col_b2 = st.columns([3, 2])

        with col_b1:
            fig_bal = px.bar(
                df_bal,
                x='Status',
                y='mean',
                text=df_bal['mean'].apply(lambda x: f"€{x:,.0f}"),
                title='Perbandingan Rata-rata Saldo Rekening (Balance)',
                color='Status',
                color_discrete_map={'Churn (Keluar)': '#EF4444', 'Stay (Bertahan)': '#10B981'}
            )
            fig_bal.update_layout(yaxis=dict(title='Rata-rata Saldo (€)'), showlegend=False, height=340)
            st.plotly_chart(fig_bal, use_container_width=True)

        with col_b2:
            st.markdown(f"""
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; border-radius:12px; padding:20px; height:100%; display:flex; flex-direction:column; justify-content:center;">
                <div style="font-size:0.85rem; font-weight:700; color:#64748B;">TOTAL SALDO HILANG DARI NASABAH CHURN</div>
                <div style="font-size:2rem; font-weight:800; color:#DC2626; margin:8px 0;">€{total_balance_at_risk:,.0f}</div>
                <div style="font-size:0.85rem; color:#475569; line-height:1.5;">
                    Nasabah yang keluar memiliki rata-rata saldo <b>€91,109</b>, signifikan lebih tinggi dibanding nasabah bertahan (<b>€72,743</b>). Ini menandakan bank mengalami <i>High-Balance Customer Drain</i>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- 4.6 UJI KEAKTIFAN ----------------
    with eda_tabs[5]:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">📌 4.6 Uji Chi-Square & Heatmap Interaksi Keaktifan vs Usia</div>
            <div class="saas-card-desc">Uji signifikansi statistik formal hubungan antara status keaktifan nasabah (IsActiveMember) dengan status Churn.</div>
        """, unsafe_allow_html=True)

        contingency_table = pd.crosstab(df_raw['IsActiveMember'], df_raw['Exited'])
        chi2, p, dof, _ = chi2_contingency(contingency_table)

        col_k1, col_k2 = st.columns([1, 2])
        with col_k1:
            st.markdown(f"""
            <div style="background:#F1F5F9; border-radius:12px; padding:18px; border:1px solid #CBD5E1;">
                <div style="font-size:0.82rem; font-weight:700; color:#475569; margin-bottom:8px;">HASIL UJI CHI-SQUARE:</div>
                <div style="font-size:1.1rem; font-weight:800; color:#0F172A;">Chi2: {chi2:.2f}</div>
                <div style="font-size:1.1rem; font-weight:800; color:#4F46E5; margin-bottom:8px;">p-value: {p:.2e}</div>
                <div style="font-size:0.82rem; color:#334155;">
                    Nilai <b>p < 0.05</b> menolak H0: Terdapat korelasi yang sangat signifikan antara keaktifan nasabah dengan keputusan churn.
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_k2:
            heatmap_data = df_raw.pivot_table(
                index='AgeGroup',
                columns='IsActiveMember',
                values='Exited',
                aggfunc=lambda x: np.round((x.sum() / len(x)) * 100, 2),
                observed=False
            ).reset_index()
            heatmap_data.columns = ['AgeGroup', 'Tidak Aktif (0)', 'Aktif (1)']

            fig_heat = px.imshow(
                heatmap_data.set_index('AgeGroup'),
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Reds",
                title="Heatmap Churn Rate (%) Berdasarkan Usia & Keaktifan"
            )
            fig_heat.update_layout(height=280)
            st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown("""
        <div style="background:#FFFBEB; border-radius:10px; padding:14px; border:1px solid #FCD34D; margin-top:8px;">
            <div style="font-weight:700; color:#92400E; margin-bottom:4px;">💡 Titik Bahaya Tertinggi (Senior & Inaktif):</div>
            <div style="font-size:0.88rem; color:#78350F;">
                Nasabah tidak aktif pada kelompok usia <b>51+ tahun</b> memiliki churn rate fantastis mencapai <b>84.94%</b>, hampir 3.5x lipat dibandingkan nasabah aktif pada usia yang sama (24.16%).
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MENU 2: PREDIKSI INDIVIDU (SIMULATOR INTERAKTIF)
# ---------------------------------------------------------
elif menu == "🎯 Prediksi Individu (Simulator)":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size:1.6rem; font-weight:800; color:#0F172A; margin-bottom:4px;">🎯 Simulator Prediksi Churn Nasabah</h2>
        <div style="color:#64748B; font-size:0.92rem;">Gunakan simulator interaktif bertenaga XGBoost untuk mengukur probabilitas nasabah menutup rekening secara akurat.</div>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("Model pipeline tidak tersedia.")
        st.stop()

    with st.form("single_pred_form"):
        st.markdown('<div class="saas-card" style="padding:16px;">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 👤 Demografi & Profil")
            geography = st.selectbox("Negara Domisili (Geography):", ["France", "Germany", "Spain"])
            gender = st.selectbox("Jenis Kelamin (Gender):", ["Female", "Male"])
            age = st.slider("Usia (Age):", 18, 92, 38, help="Usia 41+ memiliki risiko churn jauh lebih tinggi.")
            satisfaction = st.slider("Skor Kepuasan (1-5):", 1, 5, 3)

        with col2:
            st.markdown("#### 💳 Finansial & Saldo")
            credit_score = st.slider("Skor Kredit (CreditScore):", 350, 850, 650)
            balance = st.number_input("Saldo Rekening (€):", 0.0, 300000.0, 50000.0, step=2500.0, format="%.2f")
            salary = st.number_input("Estimasi Gaji Tahunan (€):", 0.0, 300000.0, 75000.0, step=2500.0, format="%.2f")
            point_earned = st.number_input("Poin Loyalitas:", 0, 1000, 450, step=25)

        with col3:
            st.markdown("#### 🤝 Produk & Interaksi")
            tenure = st.slider("Tenure (Tahun Jadi Nasabah):", 0, 10, 5)
            num_products = st.selectbox("Jumlah Produk Bank (NumOfProducts):", [1, 2, 3, 4], index=0, help="Perhatian: Produk >2 memiliki churn rate >80%!")
            card_type = st.selectbox("Tipe Kartu Kredit:", ["DIAMOND", "GOLD", "PLATINUM", "SILVER"])
            has_cr_card = st.radio("Punya Kartu Kredit?", [1, 0], format_func=lambda x: "Ya (1)" if x == 1 else "Tidak (0)", horizontal=True)
            is_active_member = st.radio("Status Keaktifan Nasabah?", [1, 0], format_func=lambda x: "Aktif (1)" if x == 1 else "Inaktif (0)", horizontal=True)

        submit_btn = st.form_submit_button("⚡ Prediksi Risiko Sekarang", use_container_width=True, type="primary")
        st.markdown('</div>', unsafe_allow_html=True)

    if submit_btn:
        input_df = pd.DataFrame([{
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': salary,
            'Satisfaction Score': satisfaction,
            'Card Type': card_type,
            'Point Earned': point_earned
        }])

        with st.spinner("Model XGBoost menganalisis indikator risiko nasabah..."):
            pred = pipeline.predict(input_df)[0]
            probs = pipeline.predict_proba(input_df)[0]
            churn_p = probs[1]
            stay_p = probs[0]

        if churn_p >= 0.60:
            box_class = "callout-high"
            title = "🚨 RISIKO TINGGI: Nasabah Berpotensi Besar Melakukan Churn"
            desc = "Pola demografis dan transaksi nasabah sangat identik dengan kelompok nasabah yang sebelumnya menutup rekening mereka."
            risk_tag = "RISIKO TINGGI"
        elif churn_p >= 0.35:
            box_class = "callout-medium"
            title = "⚠️ RISIKO MODERAT: Nasabah di Zona Waspada"
            desc = "Nasabah memiliki potensi churn moderat. Direkomendasikan melakukan engagement preventif sebelum kepuasan menurun."
            risk_tag = "RISIKO MODERAT"
        else:
            box_class = "callout-low"
            title = "✅ RISIKO RENDAH: Nasabah Diprediksi Bertahan (Loyal)"
            desc = "Nasabah menunjukkan loyalitas yang baik dan stabilitas pemanfaatan fasilitas perbankan."
            risk_tag = "RISIKO RENDAH"

        st.markdown(f"""
        <div class="callout-box {box_class}">
            <h3 style="margin:0 0 6px 0; font-weight:800;">{title}</h3>
            <p style="margin:0; font-size:0.92rem;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Metrics display
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Probabilitas Churn", f"{churn_p*100:.1f}%")
        with m2:
            st.metric("Probabilitas Loyal", f"{stay_p*100:.1f}%")
        with m3:
            st.metric("Kategori Risiko", risk_tag)
        with m4:
            st.metric("Keputusan Klasifikasi", "CHURN (1)" if pred == 1 else "STAY (0)")

        # Visual Gauge / Probability meter
        fig_prob = go.Figure(go.Bar(
            x=[churn_p * 100, stay_p * 100],
            y=['Probabilitas Churn', 'Probabilitas Loyal'],
            orientation='h',
            marker=dict(color=['#EF4444' if churn_p >= 0.5 else '#F59E0B', '#10B981']),
            text=[f"{churn_p*100:.1f}%", f"{stay_p*100:.1f}%"],
            textposition='auto',
        ))
        fig_prob.update_layout(
            title="Tingkat Keyakinan Model XGBoost (%)",
            xaxis=dict(range=[0, 100]),
            yaxis=dict(autorange="reversed"),
            height=180,
            margin=dict(l=20, r=20, t=35, b=20)
        )
        st.plotly_chart(fig_prob, use_container_width=True)

        # Personalized Action Plan based on Notebook Findings
        st.markdown("### 💡 Rekomendasi Aksi Khusus (Targeted Retention Plan)")
        actions = []
        if num_products > 2:
            actions.append(("🔴 **Jumlah Produk (>2 Produk):**", f"Nasabah memiliki {num_products} produk. Data historis membuktikan produk >2 memiliki tingkat churn ekstrem >80%.", "**Rekomendasi:** Terapkan satu paket biaya administrasi terpadu (*bundling fee*) dan tugaskan *Dedicated Relationship Manager*."))
        if geography == "Germany":
            actions.append(("🟠 **Wilayah Domisili (Jerman):**", "Nasabah berasal dari Jerman dengan rasio churn 2x lipat lebih tinggi (~32.5%).", "**Rekomendasi:** Tinjau kompetitivitas suku bunga simpanan dan berikan promosi lokal yang relevan dengan pasar Jerman."))
        if age >= 41:
            actions.append(("🟠 **Kelompok Usia (Senior 41+):**", f"Usia nasabah ({age} tahun) berada di zona churn tinggi (34%-45%).", "**Rekomendasi:** Sediakan antrean prioritas di kantor cabang dan panduan ramah lansia untuk penggunaan mobile banking."))
        if is_active_member == 0:
            actions.append(("🔴 **Status Keaktifan (Inaktif):**", "Nasabah tidak aktif bertransaksi. Nasabah inaktif usia senior memiliki churn rate hingga 84.9%.", "**Rekomendasi:** Picu notifikasi promosi re-aktivasi otomatis (cashback transaksi atau bebas biaya transfer)."))
        if satisfaction <= 2:
            actions.append(("⚠️ **Skor Kepuasan Rendah (≤2):**", f"Skor kepuasan nasabah adalah {satisfaction}/5.", "**Rekomendasi:** Lakukan penjangkauan proaktif oleh tim Customer Care untuk menyelesaikan hambatan sebelum nasabah menutup rekening."))

        if not actions:
            st.success("🎉 Nasabah ini memiliki profil yang sangat sehat dan loyal! Pertahankan dengan reward poin loyalitas rutin.")
        else:
            for title_act, desc_act, sol_act in actions:
                with st.expander(f"{title_act} Klik untuk Rekomendasi Mitigasi", expanded=True):
                    st.write(desc_act)
                    st.markdown(sol_act)

# ---------------------------------------------------------
# MENU 3: PREDIKSI MASSAL (BATCH CSV)
# ---------------------------------------------------------
elif menu == "📁 Prediksi Massal (Batch CSV)":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size:1.6rem; font-weight:800; color:#0F172A; margin-bottom:4px;">📁 Prediksi Massal Data Nasabah (Batch Upload)</h2>
        <div style="color:#64748B; font-size:0.92rem;">Unggah file CSV berisi data nasabah untuk memprediksi churn dalam jumlah besar sekaligus secara otomatis.</div>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is None:
        st.error("Model pipeline tidak tersedia.")
        st.stop()

    uploaded_file = st.file_uploader("Pilih file CSV:", type=["csv"])
    col_u1, col_u2 = st.columns([1, 4])
    with col_u1:
        use_sample = st.button("📄 Pakai Sampel Data Bawaan")

    df_b = None
    if uploaded_file is not None:
        try:
            df_b = pd.read_csv(uploaded_file)
            st.success(f"File berhasil diunggah: **{len(df_b):,}** baris data nasabah ditemukan.")
        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
    elif use_sample and df_raw is not None:
        df_b = df_raw.head(150).copy()
        st.info("Memuat 150 sampel baris data dari `customer-churn.csv`.")

    if df_b is not None:
        st.write("### Pratinjau 5 Baris Pertama:")
        st.dataframe(df_b.head(5), use_container_width=True)

        req_cols = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
            'EstimatedSalary', 'Satisfaction Score', 'Card Type', 'Point Earned'
        ]
        missing = [c for c in req_cols if c not in df_b.columns]
        if missing:
            st.error(f"File CSV kekurangan kolom wajib: {missing}")
        else:
            if st.button("🚀 Jalankan Analisis Prediksi Massal", type="primary"):
                with st.spinner("Menghitung probabilitas churn untuk seluruh data..."):
                    X_eval = df_b[req_cols].copy()
                    preds = pipeline.predict(X_eval)
                    probs = pipeline.predict_proba(X_eval)[:, 1]

                    res_df = df_b.copy()
                    res_df['Churn_Prediction'] = preds
                    res_df['Churn_Status'] = res_df['Churn_Prediction'].map({1: 'Churn', 0: 'Stay'})
                    res_df['Churn_Probability'] = np.round(probs * 100, 2)
                    res_df['Risk_Level'] = pd.cut(
                        res_df['Churn_Probability'],
                        bins=[-1, 35, 60, 101],
                        labels=['Rendah', 'Sedang', 'Tinggi']
                    )

                st.markdown("---")
                total_b = len(res_df)
                churn_b = (res_df['Churn_Prediction'] == 1).sum()
                rate_b = (churn_b / total_b) * 100
                high_b = (res_df['Risk_Level'] == 'Tinggi').sum()

                st.markdown(f"""
                <div class="kpi-container">
                    <div class="kpi-card">
                        <div class="kpi-label">Total Data Diproses</div>
                        <div class="kpi-value">{total_b:,}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Diprediksi Churn</div>
                        <div class="kpi-value" style="color:#DC2626;">{churn_b:,}</div>
                        <div class="kpi-sub"><span class="badge-danger">{rate_b:.1f}% Churn Rate</span></div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Diprediksi Bertahan</div>
                        <div class="kpi-value" style="color:#16A34A;">{total_b - churn_b:,}</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-label">Kategori Risiko Tinggi</div>
                        <div class="kpi-value" style="color:#E11D48;">{high_b:,}</div>
                        <div class="kpi-sub">Probabilitas > 60%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    fig_p = px.pie(
                        res_df,
                        names='Churn_Status',
                        title='Distribusi Prediksi (Churn vs Stay)',
                        color='Churn_Status',
                        color_discrete_map={'Churn': '#EF4444', 'Stay': '#10B981'},
                        hole=0.4
                    )
                    st.plotly_chart(fig_p, use_container_width=True)

                with col_res2:
                    fig_g = px.histogram(
                        res_df,
                        x='Geography',
                        color='Churn_Status',
                        barmode='group',
                        title='Sebaran Prediksi Berdasarkan Negara',
                        color_discrete_map={'Churn': '#EF4444', 'Stay': '#10B981'}
                    )
                    st.plotly_chart(fig_g, use_container_width=True)

                st.markdown("### 📋 Hasil Lengkap & Filter")
                filter_val = st.selectbox("Filter Tingkat Risiko:", ["Semua Data", "Risiko Tinggi Saja (>60%)", "Risiko Sedang Saja (35%-60%)", "Risiko Rendah Saja (<35%)"])
                if filter_val == "Risiko Tinggi Saja (>60%)":
                    view_df = res_df[res_df['Risk_Level'] == 'Tinggi']
                elif filter_val == "Risiko Sedang Saja (35%-60%)":
                    view_df = res_df[res_df['Risk_Level'] == 'Sedang']
                elif filter_val == "Risiko Rendah Saja (<35%)":
                    view_df = res_df[res_df['Risk_Level'] == 'Rendah']
                else:
                    view_df = res_df

                st.dataframe(view_df, use_container_width=True)

                csv_bytes = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Unduh Hasil Lengkap (.CSV)",
                    data=csv_bytes,
                    file_name="hasil_prediksi_churn_batch.csv",
                    mime="text/csv",
                    type="primary"
                )

# ---------------------------------------------------------
# MENU 4: FEATURE IMPORTANCE & STRATEGI
# ---------------------------------------------------------
elif menu == "📈 Feature Importance & Strategi":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size:1.6rem; font-weight:800; color:#0F172A; margin-bottom:4px;">📈 Feature Importance & Strategi Retensi</h2>
        <div style="color:#64748B; font-size:0.92rem;">Transparansi model XGBoost dan peta rekomendasi tindakan bisnis yang dirumuskan di notebook.</div>
    </div>
    """, unsafe_allow_html=True)

    if pipeline is not None:
        try:
            preprocessor = pipeline.named_steps['preprocessor']
            num_f = preprocessor.transformers_[0][2]
            cat_enc = preprocessor.transformers_[1][1].named_steps['encoder']
            cat_f = list(cat_enc.get_feature_names_out(preprocessor.transformers_[1][2]))
            bin_f = preprocessor.transformers_[2][2]
            all_f = num_f + cat_f + bin_f

            importances = pipeline.named_steps['classifier'].feature_importances_
            imp_df = pd.DataFrame({
                'Fitur': all_f,
                'Importance': importances
            }).sort_values(by='Importance', ascending=True)

            fig_fi = px.bar(
                imp_df,
                x='Importance',
                y='Fitur',
                orientation='h',
                title='Feature Importance Score (Best XGBoost Model)',
                color='Importance',
                color_continuous_scale='Blues'
            )
            fig_fi.update_layout(height=540, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_fi, use_container_width=True)
        except Exception as e:
            st.warning(f"Error ekstraksi feature importance: {e}")

    st.markdown("""
    <div class="saas-card">
        <div class="saas-card-title">🎯 Matriks Rekomendasi Tindakan Anti-Churn (Action Plan)</div>
        <div class="saas-card-desc">Berdasarkan hasil pemodelan dan EDA di notebook untuk meminimalkan kerugian nasabah:</div>
        
        <table style="width:100%; border-collapse:collapse; font-size:0.88rem; color:#334155;">
            <thead>
                <tr style="background:#F1F5F9; border-bottom:2px solid #CBD5E1; text-align:left;">
                    <th style="padding:10px 14px;">Segmen Nasabah</th>
                    <th style="padding:10px 14px;">Masalah Utama</th>
                    <th style="padding:10px 14px;">Rekomendasi Tindakan Konkret</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:12px 14px; font-weight:700;">Multi-Produk (>2 Produk)</td>
                    <td style="padding:12px 14px;">Kompleksitas produk & biaya administrasi menumpuk</td>
                    <td style="padding:12px 14px;">• Sediakan Personal Relationship Manager<br>• Terapkan paket bundling hemat (1 tarif terintegrasi)</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:12px 14px; font-weight:700;">Nasabah Asal Jerman</td>
                    <td style="padding:12px 14px;">Persaingan kompetitor lokal & suku bunga kurang bersaing</td>
                    <td style="padding:12px 14px;">• Benchmarking berkala suku bunga dan fitur perbankan di Jerman<br>• Rancang kampanye promosi lokal khusus Jerman</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:12px 14px; font-weight:700;">Nasabah Senior (Usia 41+)</td>
                    <td style="padding:12px 14px;">Tantangan adaptasi teknologi & kebutuhan asistensi</td>
                    <td style="padding:12px 14px;">• Buat jalur antrean asistensi prioritas di cabang<br>• Program edukasi ramah lansia untuk penggunaan m-banking</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="padding:12px 14px; font-weight:700;">Sistem Penanganan Komplain</td>
                    <td style="padding:12px 14px;">99.5% nasabah yang komplain berujung churn</td>
                    <td style="padding:12px 14px;">• Jangan tutup tiket sebelum nasabah konfirmasi puas<br>• Berikan kompensasi otomatis (poin loyalitas / bebas admin)</td>
                </tr>
                <tr>
                    <td style="padding:12px 14px; font-weight:700;">Nasabah Inaktif</td>
                    <td style="padding:12px 14px;">Hilangnya interaksi & keterikatan finansial</td>
                    <td style="padding:12px 14px;">• Manfaatkan model XGBoost sebagai sistem deteksi dini mingguan<br>• Kirim penawaran promo re-aktivasi personal otomatis</td>
                </tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# MENU 5: KAMUS DATA & INFO MODEL
# ---------------------------------------------------------
elif menu == "ℹ️ Kamus Data & Info Model":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size:1.6rem; font-weight:800; color:#0F172A; margin-bottom:4px;">ℹ️ Kamus Data & Spesifikasi Model</h2>
        <div style="color:#64748B; font-size:0.92rem;">Informasi teknis rancang bangun pipeline machine learning dan dokumentasi setiap variabel.</div>
    </div>
    """, unsafe_allow_html=True)

    col_inf1, col_inf2 = st.columns(2)
    with col_inf1:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">⚙️ Konfigurasi Arsitektur Model</div>
            <div style="font-size:0.88rem; color:#334155; line-height:1.6;">
                • <b>Model:</b> XGBoost Classifier (Tuned via GridSearch)<br>
                • <b>Parameter Terbaik:</b>
                <ul>
                    <li><code>n_estimators: 150</code></li>
                    <li><code>max_depth: 3</code></li>
                    <li><code>learning_rate: 0.1</code></li>
                    <li><code>subsample: 0.8</code></li>
                    <li><code>scale_pos_weight: ~3.91</code> (Mengatasi data imbalance 80:20)</li>
                </ul>
                • <b>Preprocessing:</b> Median Imputer + StandardScaler (Numerik), One-Hot Encoder (Kategorik), Modus Imputer (Biner).
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_inf2:
        st.markdown("""
        <div class="saas-card">
            <div class="saas-card-title">🏆 Evaluasi pada Test Set (2.000 Data)</div>
            <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                <tr style="border-bottom:1px solid #E2E8F0; padding:6px 0;">
                    <td style="font-weight:700; padding:6px 0;">ROC-AUC Score:</td>
                    <td style="color:#4F46E5; font-weight:800; text-align:right;">0.8771</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="font-weight:700; padding:6px 0;">Recall (Kelas Churn):</td>
                    <td style="color:#16A34A; font-weight:800; text-align:right;">76.0%</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="font-weight:700; padding:6px 0;">Akurasi Keseluruhan:</td>
                    <td style="font-weight:700; text-align:right;">~80.0%</td>
                </tr>
                <tr style="border-bottom:1px solid #E2E8F0;">
                    <td style="font-weight:700; padding:6px 0;">F1-Score (Churn):</td>
                    <td style="font-weight:700; text-align:right;">0.6072</td>
                </tr>
                <tr>
                    <td style="font-weight:700; padding:6px 0;">Precision (Churn):</td>
                    <td style="font-weight:700; text-align:right;">0.5057</td>
                </tr>
            </table>
            <div style="font-size:0.78rem; color:#64748B; margin-top:10px;">
                *Prioritas model adalah Recall & ROC-AUC agar tidak melewatkan nasabah yang akan churn (meminimalkan False Negative).
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="saas-card">
        <div class="saas-card-title">📖 Kamus Variabel Dataset</div>
        <table style="width:100%; border-collapse:collapse; font-size:0.88rem; color:#334155;">
            <thead>
                <tr style="background:#F1F5F9; border-bottom:2px solid #CBD5E1; text-align:left;">
                    <th style="padding:8px 12px;">Variabel</th>
                    <th style="padding:8px 12px;">Tipe</th>
                    <th style="padding:8px 12px;">Keterangan</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>CreditScore</code></td><td>Numerik</td><td>Skor kelayakan kredit nasabah (350 - 850)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>Geography</code></td><td>Kategorik</td><td>Negara domisili nasabah (France, Germany, Spain)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>Gender</code></td><td>Kategorik</td><td>Jenis kelamin nasabah (Female, Male)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>Age</code></td><td>Numerik</td><td>Usia nasabah dalam tahun</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>Tenure</code></td><td>Numerik</td><td>Lama waktu menjadi nasabah (0 - 10 tahun)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>Balance</code></td><td>Numerik</td><td>Saldo yang tersimpan dalam rekening (€)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>NumOfProducts</code></td><td>Numerik</td><td>Jumlah fasilitas/layanan produk perbankan (1 - 4)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>HasCrCard</code></td><td>Biner</td><td>Kepemilikan kartu kredit (1 = Punya, 0 = Tidak)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>IsActiveMember</code></td><td>Biner</td><td>Status keaktifan transaksi (1 = Aktif, 0 = Inaktif)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>EstimatedSalary</code></td><td>Numerik</td><td>Estimasi gaji / pendapatan tahunan (€)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>Satisfaction Score</code></td><td>Numerik</td><td>Tingkat kepuasan nasabah (1 - 5)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>Card Type</code></td><td>Kategorik</td><td>Jenis tingkatan kartu (DIAMOND, GOLD, PLATINUM, SILVER)</td></tr>
                <tr style="border-bottom:1px solid #E2E8F0;"><td style="padding:8px 12px;"><code>Point Earned</code></td><td>Numerik</td><td>Poin reward yang telah dikumpulkan nasabah</td></tr>
                <tr><td style="padding:8px 12px;"><code>Exited</code></td><td>Target</td><td>Status Churn nasabah (1 = Keluar, 0 = Bertahan)</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)
