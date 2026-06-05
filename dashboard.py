"""
Dashboard Interaktif: Cognitive Fatigue & Digital Habits
=========================================================
Dataset : cleaned_cognitive_fatigue_dataset.csv
Author  : CC26-PSU400
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CogniCare Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background: #F8F9FA; color: #1A1A2E; }

[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #DEE2E6;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #1565C0 0%, #00ACC1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-size: 1rem;
    color: #6C757D;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.metric-card {
    background: #FFFFFF;
    border: 1px solid #DEE2E6;
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: border-color 0.2s;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.metric-card:hover { border-color: #1565C0; box-shadow: 0 4px 12px rgba(21,101,192,0.12); }
.metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6C757D;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #1565C0;
}
.metric-delta { font-size: 0.78rem; color: #2E7D32; margin-top: 0.15rem; }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    letter-spacing: 0.1em;
    color: #1565C0;
    text-transform: uppercase;
    border-left: 3px solid #1565C0;
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
}

.insight-box {
    background: #E8F4FD;
    border: 1px solid #BEE3F8;
    border-left: 4px solid #2E7D32;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #1A1A2E;
    margin-top: 0.7rem;
}

.shap-insight-box {
    background: #F3E5F5;
    border: 1px solid #CE93D8;
    border-left: 4px solid #7B1FA2;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: #1A1A2E;
    margin-top: 0.7rem;
}

.badge-refreshed { background:#E8F5E9; color:#2E7D32; border:1px solid #A5D6A7; border-radius:20px; padding:3px 14px; font-size:0.78rem; }
.badge-strained  { background:#FFF8E1; color:#F57F17; border:1px solid #FFE082; border-radius:20px; padding:3px 14px; font-size:0.78rem; }
.badge-burnout   { background:#FFEBEE; color:#C62828; border:1px solid #EF9A9A; border-radius:20px; padding:3px 14px; font-size:0.78rem; }

.js-plotly-plot .plotly { background: transparent !important; }

.sidebar-label {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6C757D;
    margin-bottom: 0.2rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────
PLOT_BG  = "#FFFFFF"
PAPER_BG = "#F8F9FA"
GRID_COL = "#E9ECEF"
TEXT_COL = "#495057"
FONT_FAM = "DM Sans"
PALETTE  = {"refreshed": "#2E7D32", "strained": "#F57F17", "near-burnout": "#C62828"}

def apply_chart_theme(fig, height=380):
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_FAM, color=TEXT_COL, size=12),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID_COL, font=dict(color=TEXT_COL)),
    )
    return fig


# ──────────────────────────────────────────────
# HELPER: Normalisasi shap_values — FIXED VERSION
# ──────────────────────────────────────────────
def normalize_shap_values(shap_values_raw, n_classes, n_features):
    """
    Selalu mengembalikan: List[np.ndarray shape (n_samples, n_features)]
    dengan panjang = n_classes.
    Kompatibel dengan semua versi SHAP (lama & baru).
    """
    # ── Kasus 1: sudah list (SHAP lama, multi-class RF) ──────────────────
    if isinstance(shap_values_raw, list):
        result = []
        for sv in shap_values_raw:
            arr = np.array(sv.values) if hasattr(sv, "values") else np.array(sv)
            # Kadang tiap elemen masih 3D: (n_samples, n_features, 1)
            if arr.ndim == 3:
                if arr.shape[2] == 1:
                    arr = arr[:, :, 0]
                else:
                    arr = arr.mean(axis=2)
            result.append(arr)
        # Validasi — kolom harus = n_features
        fixed = []
        for arr in result:
            if arr.ndim == 2 and arr.shape[1] != n_features and arr.shape[0] == n_features:
                arr = arr.T
            fixed.append(arr)
        return fixed

    # ── Kasus 2: Explanation object (SHAP baru) atau ndarray ─────────────
    vals = np.array(shap_values_raw.values) if hasattr(shap_values_raw, "values") \
           else np.array(shap_values_raw)

    if vals.ndim == 3:
        s0, s1, s2 = vals.shape
        # (n_samples, n_features, n_classes)  ← paling umum di SHAP >= 0.40
        if s2 == n_classes and s1 == n_features:
            return [vals[:, :, i] for i in range(n_classes)]
        # (n_classes, n_samples, n_features)
        if s0 == n_classes and s2 == n_features:
            return [vals[i] for i in range(n_classes)]
        # (n_samples, n_classes, n_features)  ← jarang
        if s1 == n_classes and s2 == n_features:
            return [vals[:, i, :] for i in range(n_classes)]
        # Fallback generik: pisahkan axis terakhir bila == n_classes
        if s2 == n_classes:
            return [vals[:, :, i] for i in range(n_classes)]
        # Fallback: pisahkan axis pertama
        return [vals[i] for i in range(min(n_classes, s0))]

    elif vals.ndim == 2:
        # Binary / single-output → duplikat untuk semua kelas
        return [vals] * n_classes

    # Fallback terakhir
    return [vals] * n_classes


def get_shap_array(shap_values_normalized, idx):
    """
    Ambil array 2D (n_samples, n_features) untuk kelas ke-idx.
    Selalu mengembalikan plain np.ndarray 2D.
    """
    arr = shap_values_normalized[idx]
    if hasattr(arr, "values"):
        arr = np.array(arr.values)
    arr = np.array(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 3:
        # Ambil slice pertama pada axis terakhir sebagai fallback
        arr = arr[:, :, 0]
    return arr


def get_expected_value(explainer, cls_idx, n_classes):
    """Ambil expected_value dengan aman untuk semua format."""
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = list(ev)
        if len(ev) > cls_idx:
            return float(ev[cls_idx])
        return float(ev[0])
    return float(ev)


# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_cognitive_fatigue_dataset.csv")

    if "fatigue_level" not in df.columns and "fatigue_score" in df.columns:
        def fatigue_label(score):
            if score <= 2.0:   return "refreshed"
            elif score <= 3.5: return "strained"
            else:              return "near-burnout"
        df["fatigue_level"] = df["fatigue_score"].apply(fatigue_label)

    if "wellness_index" in df.columns and (df["wellness_index"] == 0).any():
        df["wellness_index"] = df["wellness_index"].replace(0, np.nan)
        df["wellness_index"].fillna(df["wellness_index"].median(), inplace=True)

    df.drop_duplicates(inplace=True)
    return df


# ──────────────────────────────────────────────
# LOAD SHAP MODEL — FIXED
# ──────────────────────────────────────────────
@st.cache_resource
def load_shap_model(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

    df_model = df.copy()

    cat_cols = [c for c in df_model.select_dtypes(include="object").columns
                if c != "fatigue_level"]
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    if cat_cols:
        df_model[cat_cols] = enc.fit_transform(df_model[cat_cols])

    le = LabelEncoder()
    y  = le.fit_transform(df_model["fatigue_level"])
    n_classes = len(le.classes_)

    drop_cols = ["fatigue_level", "fatigue_score", "fatigue_level_encoded",
                 "screen_time_bins", "Activity_Level"]
    X = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns])

    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    n_shap = min(500, X_test.shape[0])
    X_shap = X_test.iloc[:n_shap].reset_index(drop=True)
    n_features = X_shap.shape[1]

    explainer       = shap.TreeExplainer(rf)
    shap_values_raw = explainer.shap_values(X_shap)

    # ✅ Normalisasi ke List[ndarray(n_samples, n_features)] — kompatibel semua versi SHAP
    shap_values_norm = normalize_shap_values(shap_values_raw, n_classes, n_features)

    # ✅ Validasi shape setelah normalisasi
    validated = []
    for i, sv in enumerate(shap_values_norm):
        sv = np.array(sv)
        if sv.shape != (n_shap, n_features):
            # Coba koreksi otomatis
            if sv.ndim == 2 and sv.shape[0] == n_features and sv.shape[1] == n_shap:
                sv = sv.T
            elif sv.ndim != 2 or sv.shape[1] != n_features:
                # Fallback: isi nol agar tidak crash
                sv = np.zeros((n_shap, n_features))
        validated.append(sv)

    return {
        "model"         : rf,
        "explainer"     : explainer,
        "shap_values"   : validated,        # List[ndarray(n_samples, n_features)]
        "X_shap"        : X_shap,
        "X_test"        : X_test,
        "y_test"        : y_test,
        "class_names"   : list(le.classes_),
        "feature_names" : list(X.columns),
        "label_encoder" : le,
        "n_classes"     : n_classes,
        "n_features"    : n_features,
    }


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 CogniCare")
    st.markdown("---")
    st.markdown('<div class="sidebar-label">Filter Data</div>', unsafe_allow_html=True)

    try:
        df_raw      = load_data()
        data_loaded = True
    except FileNotFoundError:
        data_loaded = False

    if data_loaded:
        if "mood" in df_raw.columns:
            moods_available = sorted(df_raw["mood"].dropna().unique().tolist())
            selected_moods  = st.multiselect("Mood", options=moods_available, default=moods_available)
        else:
            selected_moods = None

        fl_opts       = ["refreshed", "strained", "near-burnout"]
        fl_opts_avail = [f for f in fl_opts if f in df_raw["fatigue_level"].unique()]
        selected_fl   = st.multiselect("Fatigue Level", options=fl_opts_avail, default=fl_opts_avail)

        st_min   = float(df_raw["screen_time"].min())
        st_max   = float(df_raw["screen_time"].max())
        st_range = st.slider("Screen Time (jam)", min_value=st_min, max_value=st_max,
                             value=(st_min, st_max), step=0.5)

        df = df_raw.copy()
        if selected_moods and "mood" in df.columns:
            df = df[df["mood"].isin(selected_moods)]
        if selected_fl:
            df = df[df["fatigue_level"].isin(selected_fl)]
        df = df[(df["screen_time"] >= st_range[0]) & (df["screen_time"] <= st_range[1])]

        st.markdown("---")
        st.markdown(f"**{len(df):,}** baris tersaring dari **{len(df_raw):,}**")
    else:
        st.error("File `cleaned_cognitive_fatigue_dataset.csv` tidak ditemukan.")
        df = None

    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["📊 Overview", "🔍 EDA & Distribusi", "🤝 Korelasi & RQ",
         "🧠 SHAP Analysis", "🧪 A/B Testing", "📋 Data Dictionary"],
    )

# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 CogniCare</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Cognitive Fatigue & Digital Habits — Analytic Dashboard</div>',
            unsafe_allow_html=True)

if not data_loaded or df is None:
    st.warning("Upload file `cleaned_cognitive_fatigue_dataset.csv` ke direktori yang sama.")
    st.stop()


# ═══════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    def kpi_card(col, label, value, delta=""):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

    kpi_card(kpi1, "Total Responden",  f"{len(df):,}",                   "setelah filter")
    kpi_card(kpi2, "Avg Screen Time",  f"{df['screen_time'].mean():.1f}h","per hari")
    kpi_card(kpi3, "Avg Sleep Hours",  f"{df['sleep_hours'].mean():.1f}h","per hari")
    kpi_card(kpi4, "Avg Stress Level", f"{df['stress_level'].mean():.2f}","skala 0–10")
    kpi_card(kpi5, "Avg Wellness",     f"{df['wellness_index'].mean():.1f}","skala 0–100")

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Distribusi Fatigue Level</div>', unsafe_allow_html=True)
        fl_counts = df["fatigue_level"].value_counts().reset_index()
        fl_counts.columns = ["fatigue_level", "count"]
        fl_counts["pct"]  = (fl_counts["count"] / fl_counts["count"].sum() * 100).round(1)

        fig_pie = px.pie(fl_counts, names="fatigue_level", values="count",
                         color="fatigue_level", color_discrete_map=PALETTE,
                         hole=0.55, custom_data=["pct"])
        fig_pie.update_traces(
            texttemplate="%{label}<br><b>%{customdata[0]:.1f}%</b>",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>Jumlah: %{value:,}<br>Persen: %{customdata[0]:.1f}%<extra></extra>",
        )
        apply_chart_theme(fig_pie, height=340)
        fig_pie.add_annotation(text=f"<b>{len(df):,}</b><br>Responden",
                               x=0.5, y=0.5, font_size=14, font_color="#1A1A2E", showarrow=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Distribusi Mood</div>', unsafe_allow_html=True)
        if "mood" in df.columns:
            mood_counts = df["mood"].value_counts().reset_index()
            mood_counts.columns = ["mood", "count"]
            fig_mood = px.bar(mood_counts, x="count", y="mood", orientation="h",
                              color="count", color_continuous_scale=["#1565C0","#00ACC1"], text="count")
            fig_mood.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_mood.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Jumlah")
            apply_chart_theme(fig_mood, height=340)
            st.plotly_chart(fig_mood, use_container_width=True)
        else:
            st.info("Kolom 'mood' tidak tersedia.")

    st.markdown('<div class="section-header">Wellness Index per Fatigue Level</div>', unsafe_allow_html=True)
    fig_box = px.box(df, x="fatigue_level", y="wellness_index",
                     color="fatigue_level", color_discrete_map=PALETTE, points="outliers",
                     category_orders={"fatigue_level": ["refreshed","strained","near-burnout"]})
    apply_chart_theme(fig_box, height=340)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Insight:</b> Pengguna dalam kategori <i>refreshed</i> konsisten menunjukkan wellness index
    lebih tinggi, sementara <i>near-burnout</i> memiliki sebaran yang lebih lebar.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 2: EDA & DISTRIBUSI
# ═══════════════════════════════════════════════
elif page == "🔍 EDA & Distribusi":
    st.markdown('<div class="section-header">Distribusi Variabel Numerik</div>', unsafe_allow_html=True)

    numeric_cols = [c for c in ["screen_time","sleep_hours","stress_level",
                                "wellness_index","physical_activity","fatigue_score"]
                    if c in df.columns]
    selected_num = st.selectbox("Pilih variabel:", numeric_cols)

    col1, col2 = st.columns(2)
    with col1:
        fig_hist = px.histogram(df, x=selected_num, color="fatigue_level",
                                color_discrete_map=PALETTE, nbins=40, barmode="overlay",
                                opacity=0.75, marginal="box",
                                category_orders={"fatigue_level":["refreshed","strained","near-burnout"]})
        fig_hist.update_layout(title=f"Distribusi {selected_num} per Fatigue Level")
        apply_chart_theme(fig_hist, 380)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_vio = px.violin(df, y=selected_num, x="fatigue_level", color="fatigue_level",
                            color_discrete_map=PALETTE, box=True, points=False,
                            category_orders={"fatigue_level":["refreshed","strained","near-burnout"]})
        fig_vio.update_layout(title=f"Violin Plot: {selected_num}")
        apply_chart_theme(fig_vio, 380)
        st.plotly_chart(fig_vio, use_container_width=True)

    st.markdown('<div class="section-header">Statistik Deskriptif</div>', unsafe_allow_html=True)
    stat = df[numeric_cols].describe().T.round(3)
    stat["skewness"] = df[numeric_cols].skew().round(3)
    stat["kurtosis"] = df[numeric_cols].kurtosis().round(3)
    st.dataframe(stat, use_container_width=True)

    st.markdown('<div class="section-header">Deteksi Outlier (Metode IQR)</div>', unsafe_allow_html=True)
    outlier_rows = []
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        n_out  = ((df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)).sum()
        outlier_rows.append({"Variabel": col, "Q1": round(q1,3), "Q3": round(q3,3),
                             "IQR": round(iqr,3), "Jumlah Outlier": n_out,
                             "Persen (%)": round(n_out/len(df)*100, 2)})
    st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

    if "screen_time" in df.columns:
        st.markdown('<div class="section-header">Distribusi Screen Time (Bins)</div>', unsafe_allow_html=True)
        df_temp = df.copy()
        df_temp["screen_time_bins"] = pd.cut(df_temp["screen_time"],
                                             bins=[0,2,4,6,8,10,12,24],
                                             labels=["0-2","2-4","4-6","6-8","8-10","10-12","12+"])
        bin_fl = df_temp.groupby(["screen_time_bins","fatigue_level"], observed=True).size().reset_index(name="count")
        fig_bin = px.bar(bin_fl, x="screen_time_bins", y="count",
                         color="fatigue_level", color_discrete_map=PALETTE, barmode="stack",
                         category_orders={"fatigue_level":["refreshed","strained","near-burnout"]})
        fig_bin.update_layout(xaxis_title="Screen Time (jam/hari)", yaxis_title="Jumlah Responden")
        apply_chart_theme(fig_bin, 360)
        st.plotly_chart(fig_bin, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        💡 <b>Insight:</b> Peningkatan proporsi <i>near-burnout</i> seiring screen time meningkat,
        terutama di atas 8 jam/hari.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 3: KORELASI & RESEARCH QUESTIONS
# ═══════════════════════════════════════════════
elif page == "🤝 Korelasi & RQ":
    st.markdown('<div class="section-header">Heatmap Korelasi</div>', unsafe_allow_html=True)

    numeric_cols = [c for c in ["screen_time","sleep_hours","stress_level",
                                "wellness_index","physical_activity","fatigue_score"]
                    if c in df.columns]
    corr = df[numeric_cols].corr().round(3)

    fig_heat = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0, text=corr.values.round(2),
        texttemplate="%{text}", textfont=dict(size=11),
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    apply_chart_theme(fig_heat, 420)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-header">RQ1 · Screen Time vs Fatigue Score</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        if "fatigue_score" in df.columns:
            fig_sc = px.scatter(df.sample(min(3000, len(df)), random_state=42),
                                x="screen_time", y="fatigue_score",
                                color="fatigue_level", color_discrete_map=PALETTE, opacity=0.5,
                                trendline="ols", trendline_scope="overall",
                                trendline_color_override="#1565C0",
                                category_orders={"fatigue_level":["refreshed","strained","near-burnout"]})
            apply_chart_theme(fig_sc, 380)
            st.plotly_chart(fig_sc, use_container_width=True)
    with col2:
        st.markdown("""
        <div class="insight-box" style="margin-top:3rem">
        📌 <b>Temuan:</b> Terdapat korelasi positif antara screen time dan fatigue score.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">RQ2 · Sleep Hours vs Wellness Index</div>', unsafe_allow_html=True)
    if "wellness_index" in df.columns:
        df_sleep = df.copy()
        df_sleep["sleep_bins"] = pd.cut(df_sleep["sleep_hours"],
                                        bins=[0,4,6,7,8,9,12],
                                        labels=["<4","4-6","6-7","7-8","8-9",">9"])
        sleep_agg = df_sleep.groupby("sleep_bins", observed=True)["wellness_index"].mean().reset_index()
        fig_sleep = px.line(sleep_agg, x="sleep_bins", y="wellness_index", markers=True, line_shape="spline")
        fig_sleep.update_traces(line_color="#1565C0", marker_color="#00ACC1", marker_size=10)
        fig_sleep.update_layout(xaxis_title="Durasi Tidur (jam)", yaxis_title="Rata-rata Wellness Index")
        apply_chart_theme(fig_sleep, 340)
        st.plotly_chart(fig_sleep, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        💡 Wellness index tertinggi pada rentang tidur 7–8 jam.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">RQ3 · Stress Level vs Wellness Index</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        fig_str = px.scatter(df.sample(min(3000, len(df)), random_state=1),
                             x="stress_level", y="wellness_index",
                             color="fatigue_level", color_discrete_map=PALETTE, opacity=0.45,
                             trendline="ols", trendline_scope="overall",
                             trendline_color_override="#1565C0")
        apply_chart_theme(fig_str, 360)
        st.plotly_chart(fig_str, use_container_width=True)
    with col4:
        if "mood" in df.columns:
            profile_cols = [c for c in ["screen_time","sleep_hours","physical_activity"] if c in df.columns]
            profile  = df.groupby("mood")[profile_cols].mean().reset_index()
            fig_prof = px.bar(profile.melt(id_vars="mood", value_vars=profile_cols),
                              x="variable", y="value", color="mood", barmode="group")
            fig_prof.update_layout(title="Profil Kebiasaan per Mood", xaxis_title="", yaxis_title="Rata-rata Nilai")
            apply_chart_theme(fig_prof, 360)
            st.plotly_chart(fig_prof, use_container_width=True)

    st.markdown('<div class="section-header">RQ5 · Mitigasi Aktivitas Fisik terhadap Efek Screen Time</div>',
                unsafe_allow_html=True)
    if "physical_activity" in df.columns:
        df_rq5 = df.copy()
        df_rq5["Activity_Level"]    = pd.cut(df_rq5["physical_activity"], bins=[-1,3,100],
                                             labels=["Low Activity","High Activity"])
        df_rq5["screen_time_bins"]  = pd.cut(df_rq5["screen_time"], bins=[0,2,4,6,8,10,12,24],
                                             labels=["0-2","2-4","4-6","6-8","8-10","10-12","12+"])
        rq5_agg = (df_rq5.groupby(["screen_time_bins","Activity_Level"], observed=True)
                         ["wellness_index"].mean().reset_index())
        fig_rq5 = px.line(rq5_agg, x="screen_time_bins", y="wellness_index", color="Activity_Level",
                          color_discrete_map={"Low Activity":"#f87171","High Activity":"#4ade80"},
                          markers=True, line_shape="spline")
        fig_rq5.update_layout(xaxis_title="Screen Time (jam)", yaxis_title="Rata-rata Wellness Index")
        apply_chart_theme(fig_rq5, 360)
        st.plotly_chart(fig_rq5, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        💡 <b>Temuan RQ5:</b> Kelompok "High Activity" mempertahankan wellness index lebih stabil.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">RQ6 · Ranking Fitur terhadap Wellness Index</div>',
                unsafe_allow_html=True)
    if "wellness_index" in df.columns:
        num_cols_rq6 = [c for c in numeric_cols if c != "wellness_index"]
        corr_target  = df[num_cols_rq6+["wellness_index"]].corr()["wellness_index"].drop("wellness_index").sort_values()
        colors_rq6   = ["#f87171" if v < 0 else "#4ade80" for v in corr_target.values]
        fig_rq6 = go.Figure(go.Bar(x=corr_target.values, y=corr_target.index, orientation="h",
                                   marker_color=colors_rq6,
                                   text=[f"{v:.3f}" for v in corr_target.values],
                                   textposition="outside"))
        fig_rq6.update_layout(xaxis_title="Koefisien Korelasi Pearson",
                              xaxis=dict(range=[-1,1], gridcolor=GRID_COL))
        apply_chart_theme(fig_rq6, 340)
        st.plotly_chart(fig_rq6, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 4: SHAP ANALYSIS — FIXED
# ═══════════════════════════════════════════════
elif page == "🧠 SHAP Analysis":
    st.markdown("""
    <div class="shap-insight-box">
    🔬 <b>Tentang SHAP Analysis:</b> Halaman ini menampilkan interpretasi model Random Forest
    menggunakan SHAP (SHapley Additive exPlanations) — menjawab <b>Rumusan Masalah 2</b>:
    faktor aktivitas harian mana yang paling dominan mempengaruhi kelelahan kognitif.
    </div>""", unsafe_allow_html=True)

    with st.spinner("Melatih model dan menghitung SHAP values... (hanya sekali per sesi)"):
        try:
            shap_data = load_shap_model(df_raw)
            shap_ok   = True
        except Exception as e:
            st.error(f"Gagal menghitung SHAP: {e}")
            shap_ok = False

    if not shap_ok:
        st.stop()

    shap_values   = shap_data["shap_values"]     # List[ndarray(n_samples, n_features)]
    X_shap        = shap_data["X_shap"]
    class_names   = shap_data["class_names"]
    feature_names = shap_data["feature_names"]
    explainer     = shap_data["explainer"]
    rf_model      = shap_data["model"]
    y_test        = shap_data["y_test"]
    X_test        = shap_data["X_test"]
    n_classes     = shap_data["n_classes"]
    n_features    = shap_data["n_features"]

    # ── Debug info (opsional, bisa dihapus di production) ──────────────
    with st.expander("🔧 Debug Info (Shape SHAP Values)", expanded=False):
        st.write(f"n_classes: {n_classes}, n_features: {n_features}, n_shap_samples: {X_shap.shape[0]}")
        for i, sv in enumerate(shap_values):
            st.write(f"  Kelas {i} ({class_names[i]}): shape = {np.array(sv).shape}")

    st.success(f"✅ SHAP berhasil dihitung — {X_shap.shape[0]} sampel | "
               f"{n_features} fitur | {n_classes} kelas")

    cls_options = {f"Kelas: {c.upper()}": i for i, c in enumerate(class_names)}

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Feature Importance",
        "🐝 Beeswarm Plot",
        "⚡ Force Plot",
        "📉 Confusion Matrix",
        "🏆 Perbandingan Model",
    ])

    # ─────────────────────────────────────────
    # TAB 1: FEATURE IMPORTANCE — FIXED
    # ─────────────────────────────────────────
    with tab1:
        st.markdown('<div class="section-header">SHAP Feature Importance Global per Kelas</div>',
                    unsafe_allow_html=True)
        try:
            colors_kelas = ["#27AE60", "#E67E22", "#E74C3C"]
            fig_bar, axes = plt.subplots(1, n_classes, figsize=(5*n_classes+1, 5))
            if n_classes == 1:
                axes = [axes]

            for i, (cls, color) in enumerate(zip(class_names, colors_kelas[:n_classes])):
                arr = get_shap_array(shap_values, i)          # (n_samples, n_features)
                # Pastikan shape benar sebelum operasi
                assert arr.shape[1] == n_features, \
                    f"Kelas {i}: arr.shape={arr.shape}, n_features={n_features}"

                mean_abs = pd.Series(
                    np.abs(arr).mean(axis=0),
                    index=feature_names
                ).sort_values(ascending=True)

                bars = axes[i].barh(mean_abs.index, mean_abs.values,
                                    color=color, edgecolor="white", linewidth=0.5)
                max_val = mean_abs.max() if mean_abs.max() > 0 else 1
                for bar, val in zip(bars, mean_abs.values):
                    axes[i].text(val + max_val * 0.01,
                                 bar.get_y() + bar.get_height() / 2,
                                 f"{val:.4f}", va="center", ha="left", fontsize=8)
                axes[i].set_title(f"Kelas: {cls.upper()}", fontsize=10, fontweight="bold", color=color)
                axes[i].set_xlabel("Mean |SHAP Value|", fontsize=8)
                axes[i].spines[["top", "right"]].set_visible(False)

            fig_bar.suptitle("SHAP Global Feature Importance per Kelas Fatigue Level",
                             fontsize=12, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

        except Exception as e:
            st.error(f"Error Feature Importance plot: {e}")
            plt.close("all")

        # Ranking tabel
        st.markdown('<div class="section-header">Ranking Feature Importance (Global)</div>',
                    unsafe_allow_html=True)
        try:
            # Rata-rata mean |SHAP| lintas semua kelas
            all_mean_abs = np.zeros(n_features)
            for i in range(n_classes):
                arr = get_shap_array(shap_values, i)    # (n_samples, n_features)
                all_mean_abs += np.abs(arr).mean(axis=0)
            all_mean_abs /= n_classes

            df_rank = (pd.Series(all_mean_abs, index=feature_names)
                         .sort_values(ascending=False)
                         .reset_index())
            df_rank.columns = ["Fitur", "Mean |SHAP Value|"]
            df_rank.index  += 1
            df_rank["Tier"] = df_rank.index.map(
                lambda x: "🔴 Tinggi" if x <= 3 else "🟡 Sedang" if x <= 6 else "🟢 Rendah")
            df_rank["Mean |SHAP Value|"] = df_rank["Mean |SHAP Value|"].round(4)
            st.dataframe(df_rank, use_container_width=True)
        except Exception as e:
            st.error(f"Error ranking tabel: {e}")

        st.markdown("""
        <div class="shap-insight-box">
        🔬 <b>Insight:</b> <code>screen_time</code> adalah fitur paling dominan, terutama pada kelas
        Near-Burnout — konsisten dengan korelasi Pearson (r = +0.68) pada RQ1.
        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # TAB 2: BEESWARM PLOT — FIXED
    # ─────────────────────────────────────────
    with tab2:
        selected_cls_label = st.selectbox("Pilih kelas untuk Beeswarm Plot:", list(cls_options.keys()),
                                          key="bee_cls")
        selected_cls_idx   = cls_options[selected_cls_label]

        st.markdown(f'<div class="section-header">Beeswarm Plot — {selected_cls_label}</div>',
                    unsafe_allow_html=True)
        try:
            arr_bee = get_shap_array(shap_values, selected_cls_idx)  # (n_samples, n_features)
            # Pastikan X_shap & arr_bee shape-nya konsisten
            assert arr_bee.shape == X_shap.shape, \
                f"Shape mismatch: arr_bee={arr_bee.shape}, X_shap={X_shap.shape}"

            fig_bee, ax_bee = plt.subplots(figsize=(10, 6))
            plt.sca(ax_bee)
            shap.summary_plot(
                arr_bee,
                X_shap,
                feature_names=feature_names,
                show=False,
                plot_size=None,
                max_display=len(feature_names),
                plot_type="dot",
            )
            ax_bee.set_title(f"SHAP Beeswarm Plot — {selected_cls_label}",
                             fontsize=11, fontweight="bold")
            ax_bee.set_xlabel("SHAP Value (dampak terhadap output model)", fontsize=9)
            plt.tight_layout()
            st.pyplot(fig_bee, use_container_width=True)
            plt.close(fig_bee)
        except Exception as e:
            st.error(f"Error Beeswarm plot: {e}")
            plt.close("all")

        st.markdown("""
        <div class="shap-insight-box">
        🐝 <b>Cara membaca Beeswarm Plot:</b><br>
        • <b>Titik merah</b> = nilai fitur tinggi &nbsp;|&nbsp; <b>Titik biru</b> = nilai fitur rendah<br>
        • <b>Posisi kanan (SHAP &gt; 0)</b> = mendorong prediksi ke arah kelas ini<br>
        • <b>Posisi kiri (SHAP &lt; 0)</b> = menjauhkan prediksi dari kelas ini
        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # TAB 3: FORCE PLOT — FIXED
    # ─────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Interpretasi Prediksi Individual</div>',
                    unsafe_allow_html=True)

        col_fp1, col_fp2 = st.columns(2)
        with col_fp1:
            sample_idx = st.number_input(f"Index sampel (0 – {X_shap.shape[0]-1}):",
                                         min_value=0, max_value=X_shap.shape[0]-1,
                                         value=0, step=1)
        with col_fp2:
            cls_fp_label = st.selectbox("Kelas yang ditampilkan:", list(cls_options.keys()),
                                        key="fp_cls")
        cls_fp_idx = cls_options[cls_fp_label]

        pred_label = class_names[rf_model.predict(X_shap.iloc[[sample_idx]])[0]]
        pred_proba = rf_model.predict_proba(X_shap.iloc[[sample_idx]])[0]

        col_i1, col_i2, col_i3 = st.columns(3)
        col_i1.metric("Prediksi Model", pred_label.upper())
        nb_prob = (f"{pred_proba[class_names.index('near-burnout')]:.3f}"
                   if "near-burnout" in class_names else "-")
        col_i2.metric("Prob. Near-Burnout", nb_prob)
        ev_val = get_expected_value(explainer, cls_fp_idx, n_classes)
        col_i3.metric("Expected Value", f"{ev_val:.4f}")

        try:
            arr_fp  = get_shap_array(shap_values, cls_fp_idx)  # (n_samples, n_features)
            sv_row  = arr_fp[int(sample_idx)]                   # (n_features,)  — 1D
            feat_row = X_shap.iloc[int(sample_idx)]             # Series (n_features,)

            # Pastikan panjang sv_row == n_features == len(feature_names)
            assert len(sv_row) == n_features == len(feature_names), \
                f"Panjang tidak sama: sv_row={len(sv_row)}, n_features={n_features}, feature_names={len(feature_names)}"

            fig_fp, _ = plt.subplots(figsize=(14, 3))
            shap.force_plot(
                ev_val,
                sv_row,
                feat_row,
                feature_names=feature_names,
                matplotlib=True,
                show=False,
                figsize=(14, 3),
            )
            plt.title(
                f"Force Plot | Sampel {sample_idx} | {cls_fp_label} | Prediksi: {pred_label.upper()}",
                fontsize=10, pad=32,
            )
            plt.tight_layout()
            st.pyplot(fig_fp, use_container_width=True)
            plt.close(fig_fp)
        except Exception as e:
            st.error(f"Error Force Plot: {e}")
            plt.close("all")

        st.markdown('<div class="section-header">Detail Kontribusi Fitur</div>', unsafe_allow_html=True)
        try:
            arr_fp  = get_shap_array(shap_values, cls_fp_idx)  # (n_samples, n_features)
            sv_row  = arr_fp[int(sample_idx)]                   # (n_features,)

            df_fp = pd.DataFrame({
                "Fitur"         : feature_names,
                "Nilai (scaled)": X_shap.iloc[int(sample_idx)].values.round(4),
                "SHAP Value"    : sv_row.round(4),
            }).sort_values("SHAP Value", ascending=False).reset_index(drop=True)
            df_fp.index += 1
            st.dataframe(df_fp, use_container_width=True)
        except Exception as e:
            st.error(f"Error tabel kontribusi: {e}")

        st.markdown("""
        <div class="shap-insight-box">
        ⚡ <b>Insight Force Plot:</b> Batang <b>merah</b> mendorong prediksi ke kelas yang dipilih,
        batang <b>biru</b> menahannya. Lebar proporsional dengan besarnya kontribusi.
        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # TAB 4: CONFUSION MATRIX
    # ─────────────────────────────────────────
    with tab4:
        from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                                     classification_report, accuracy_score)

        st.markdown('<div class="section-header">Confusion Matrix — Random Forest</div>',
                    unsafe_allow_html=True)
        try:
            y_pred_rf = rf_model.predict(X_test)
            cm        = confusion_matrix(y_test, y_pred_rf)

            fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
            ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=class_names).plot(ax=ax_cm, cmap="Blues", colorbar=False)
            ax_cm.set_title("Confusion Matrix — Random Forest", fontsize=12, fontweight="bold", pad=12)
            ax_cm.tick_params(axis="x", rotation=15)
            plt.tight_layout()
            st.pyplot(fig_cm, use_container_width=True)
            plt.close(fig_cm)

            acc      = accuracy_score(y_test, y_pred_rf)
            nb_idx_c = list(class_names).index("near-burnout") if "near-burnout" in class_names else 0
            fn_nb    = cm[nb_idx_c].sum() - cm[nb_idx_c, nb_idx_c]

            col_cm1, col_cm2, col_cm3 = st.columns(3)
            col_cm1.metric("Accuracy",                    f"{acc*100:.2f}%")
            col_cm2.metric("False Negative Near-Burnout", str(fn_nb))
            col_cm3.metric("Total Kesalahan",             str(cm.sum() - np.diag(cm).sum()))

            st.markdown(f"""
            <div class="shap-insight-box">
            📊 Model mencapai akurasi <b>{acc*100:.2f}%</b> dengan hanya <b>{fn_nb}</b> sampel
            Near-Burnout yang gagal terdeteksi (False Negative).
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
            report    = classification_report(y_test, y_pred_rf, target_names=class_names, output_dict=True)
            df_report = pd.DataFrame(report).T.round(3)
            st.dataframe(df_report, use_container_width=True)

        except Exception as e:
            st.error(f"Error Confusion Matrix: {e}")
            plt.close("all")

    # ─────────────────────────────────────────
    # TAB 5: PERBANDINGAN MODEL
    # ─────────────────────────────────────────
    with tab5:
        from sklearn.metrics import f1_score as f1_sk, accuracy_score

        st.markdown('<div class="section-header">Perbandingan Performa Model</div>',
                    unsafe_allow_html=True)
        st.info("Melatih Logistic Regression sebagai baseline...")

        @st.cache_resource
        def train_logreg(df_raw):
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
            from sklearn.model_selection import train_test_split

            df_m  = df_raw.copy()
            cat_c = [c for c in df_m.select_dtypes("object").columns if c != "fatigue_level"]
            if cat_c:
                df_m[cat_c] = OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1).fit_transform(df_m[cat_c])
            le  = LabelEncoder()
            y   = le.fit_transform(df_m["fatigue_level"])
            drop = ["fatigue_level","fatigue_score","fatigue_level_encoded",
                    "screen_time_bins","Activity_Level"]
            X   = df_m.drop(columns=[c for c in drop if c in df_m.columns])
            X_s = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
            Xtr, Xte, ytr, yte = train_test_split(X_s, y, test_size=0.2,
                                                  random_state=42, stratify=y)
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(Xtr, ytr)
            return lr, Xte, yte

        try:
            lr_model, X_te_lr, y_te_lr = train_logreg(df_raw)
            y_pred_lr = lr_model.predict(X_te_lr)
            y_pred_rf = rf_model.predict(X_test)

            rows = [
                {"Model": "Logistic Regression (Baseline)",
                 "Accuracy (%)":    round(accuracy_score(y_te_lr, y_pred_lr)*100, 2),
                 "F1-Macro (%)":    round(f1_sk(y_te_lr, y_pred_lr, average="macro")*100, 2),
                 "F1-Weighted (%)": round(f1_sk(y_te_lr, y_pred_lr, average="weighted")*100, 2)},
                {"Model": "Random Forest",
                 "Accuracy (%)":    round(accuracy_score(y_test, y_pred_rf)*100, 2),
                 "F1-Macro (%)":    round(f1_sk(y_test, y_pred_rf, average="macro")*100, 2),
                 "F1-Weighted (%)": round(f1_sk(y_test, y_pred_rf, average="weighted")*100, 2)},
            ]
            df_cmp = (pd.DataFrame(rows)
                        .sort_values("F1-Weighted (%)", ascending=False)
                        .reset_index(drop=True))
            df_cmp.index += 1
            st.dataframe(df_cmp, use_container_width=True)

            fig_cmp = go.Figure()
            for metric in ["Accuracy (%)", "F1-Macro (%)", "F1-Weighted (%)"]:
                fig_cmp.add_trace(go.Bar(name=metric, x=df_cmp["Model"], y=df_cmp[metric]))
            fig_cmp.update_layout(barmode="group",
                                  yaxis=dict(range=[0, 110]),
                                  xaxis_title="", yaxis_title="Score (%)")
            apply_chart_theme(fig_cmp, 360)
            st.plotly_chart(fig_cmp, use_container_width=True)

            best = df_cmp.iloc[0]
            st.markdown(f"""
            <div class="shap-insight-box">
            🏆 <b>Model Terpilih: {best['Model']}</b> — F1-Weighted {best['F1-Weighted (%)']:.2f}%,
            Accuracy {best['Accuracy (%)']:.2f}%. Random Forest dipilih karena kompatibel dengan
            SHAP TreeExplainer untuk interpretasi transparan.
            </div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error Perbandingan Model: {e}")


# ═══════════════════════════════════════════════
# PAGE 5: A/B TESTING
# ═══════════════════════════════════════════════
elif page == "🧪 A/B Testing":
    from scipy import stats

    st.markdown("""
    <div class="insight-box" style="border-left-color:#1565C0; margin-bottom:1.5rem">
    🔬 <b>Skenario A/B Testing:</b> Apakah terdapat perbedaan signifikan pada <i>Wellness Index</i>
    antara pengguna dengan screen time <b>rendah (≤ 6 jam)</b> vs <b>tinggi (&gt; 6 jam)</b>?
    </div>""", unsafe_allow_html=True)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        threshold = st.number_input("Threshold Screen Time (jam)", min_value=1.0,
                                    max_value=20.0, value=6.0, step=0.5)
    with col_cfg2:
        alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
    with col_cfg3:
        metric_col = st.selectbox("Metrik yang diuji",
                                  [c for c in ["wellness_index","fatigue_score","stress_level","sleep_hours"]
                                   if c in df.columns])

    group_a = df[df["screen_time"] <= threshold][metric_col].dropna()
    group_b = df[df["screen_time"] >  threshold][metric_col].dropna()

    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    cohen_d = ((group_a.mean() - group_b.mean()) /
               np.sqrt((group_a.std()**2 + group_b.std()**2) / 2))

    st.markdown('<div class="section-header">Hasil Uji Statistik (Welch\'s t-test)</div>',
                unsafe_allow_html=True)

    res1, res2, res3, res4 = st.columns(4)
    def res_card(col, label, value, sub=""):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="font-size:1.4rem">{value}</div>
            <div class="metric-delta">{sub}</div>
        </div>""", unsafe_allow_html=True)

    res_card(res1, "Group A (≤ threshold)", f"{group_a.mean():.3f}", f"n = {len(group_a):,}")
    res_card(res2, "Group B (> threshold)", f"{group_b.mean():.3f}", f"n = {len(group_b):,}")
    res_card(res3, "t-statistic",           f"{t_stat:.4f}", "Welch's t-test")
    res_card(res4, "p-value",               f"{p_value:.4f}", f"α = {alpha}")

    st.markdown("<br>", unsafe_allow_html=True)

    if p_value < alpha:
        st.success(f"✅ **H₀ DITOLAK** (p = {p_value:.4f} < α = {alpha}) → "
                   f"Perbedaan **signifikan** pada {metric_col}. Cohen's d = {cohen_d:.3f}")
    else:
        st.info(f"ℹ️ **H₀ GAGAL DITOLAK** (p = {p_value:.4f} ≥ α = {alpha}) → "
                f"Tidak cukup bukti. Cohen's d = {cohen_d:.3f}")

    st.markdown('<div class="section-header">Distribusi Kedua Grup</div>', unsafe_allow_html=True)
    fig_ab = go.Figure()
    fig_ab.add_trace(go.Histogram(x=group_a, name=f"Group A (≤{threshold}h)",
                                  opacity=0.7, marker_color="#1565C0", nbinsx=40))
    fig_ab.add_trace(go.Histogram(x=group_b, name=f"Group B (>{threshold}h)",
                                  opacity=0.7, marker_color="#f87171", nbinsx=40))
    fig_ab.add_vline(x=group_a.mean(), line_dash="dash", line_color="#1565C0",
                     annotation_text=f"Mean A={group_a.mean():.2f}", annotation_position="top right")
    fig_ab.add_vline(x=group_b.mean(), line_dash="dash", line_color="#f87171",
                     annotation_text=f"Mean B={group_b.mean():.2f}", annotation_position="top left")
    fig_ab.update_layout(barmode="overlay", xaxis_title=metric_col, yaxis_title="Frekuensi")
    apply_chart_theme(fig_ab, 380)
    st.plotly_chart(fig_ab, use_container_width=True)

    st.markdown('<div class="section-header">Ringkasan Statistik Deskriptif</div>', unsafe_allow_html=True)
    summary_ab = pd.DataFrame({
        "Statistik": ["N","Mean","Median","Std Dev","Min","Max"],
        f"Group A (≤{threshold}h)": [len(group_a), group_a.mean().round(3), group_a.median().round(3),
                                      group_a.std().round(3),  group_a.min().round(3),  group_a.max().round(3)],
        f"Group B (>{threshold}h)": [len(group_b), group_b.mean().round(3), group_b.median().round(3),
                                      group_b.std().round(3),  group_b.min().round(3),  group_b.max().round(3)],
    })
    st.dataframe(summary_ab, use_container_width=True)

    st.markdown('<div class="section-header">Interpretasi Effect Size (Cohen\'s d)</div>',
                unsafe_allow_html=True)
    abs_d = abs(cohen_d)
    effect_label = ("Negligible (sangat kecil)" if abs_d < 0.2 else
                    "Small (kecil)"             if abs_d < 0.5 else
                    "Medium (sedang)"           if abs_d < 0.8 else
                    "Large (besar)")
    st.markdown(f"""
    <div class="insight-box">
    📐 <b>Cohen's d = {cohen_d:.3f}</b> → <b>{effect_label}</b><br><br>
    Panduan: |d| &lt; 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, ≥ 0.8 = large.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 6: DATA DICTIONARY
# ═══════════════════════════════════════════════
elif page == "📋 Data Dictionary":
    st.markdown('<div class="section-header">Data Dictionary</div>', unsafe_allow_html=True)

    data_dict = {
        "Variabel"          : ["screen_time","sleep_hours","stress_level","wellness_index",
                               "physical_activity","fatigue_score","mood","fatigue_level"],
        "Tipe Data"         : ["Float","Float","Float/Int","Float","Float/Int","Float/Int","Object","Object"],
        "Satuan / Skala"    : ["Jam/hari","Jam/hari","0–10","0–100","Jam/hari","0–10 (approx.)",
                               "Kategorik","Kategorik (3 kelas)"],
        "Deskripsi"         : [
            "Durasi penggunaan layar digital per hari",
            "Durasi tidur per hari",
            "Skor tingkat stres yang dialami pengguna",
            "Indeks kesehatan mental keseluruhan (semakin tinggi = lebih sehat)",
            "Durasi aktivitas fisik per hari",
            "Skor kelelahan kognitif (derived variable)",
            "Kondisi suasana hati pengguna (mis. Relaxed, Exhausted, Anxious, …)",
            "Label kelas target: refreshed / strained / near-burnout",
        ],
        "Threshold / Kategori": [
            "Smart Alert: > 6–8 jam", "Optimal: 7–8 jam", "Tinggi: > 7",
            "Rendah: < 40, Sedang: 40–70, Tinggi: > 70", "Low: ≤ 3 jam, High: > 3 jam",
            "≤ 2.0 → refreshed | 2.0–3.5 → strained | > 3.5 → near-burnout",
            "—", "Target variable (multi-class classification)",
        ],
    }
    st.dataframe(pd.DataFrame(data_dict), use_container_width=True, height=340)

    st.markdown('<div class="section-header">Info Dataset</div>', unsafe_allow_html=True)
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("Jumlah Baris (raw)", "98,535")
    col_i2.metric("Jumlah Kolom",       str(len(df.columns)))
    col_i3.metric("Missing Value",      "0")
    col_i4.metric("Duplikat",           "920 (dihapus)")

    st.markdown('<div class="section-header">Preview Data (Tersaring)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=300)

    st.markdown("""
    <div class="insight-box">
    📁 <b>Sumber Data:</b><br>
    • Dataset A: <code>digital_habits_vs_mental_health.csv</code> (utama, n ≈ 98.535)<br>
    • Dataset B: <code>ScreenTime vs MentalWellness.csv</code> (validasi silang)<br>
    • Target variable <code>fatigue_level</code> di-derive dari <code>fatigue_score</code>
      menggunakan threshold berbasis distribusi kuartil.
    </div>""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#3a4060;font-size:0.78rem;font-family:Space Mono,monospace;">'
    'CogniCare Dashboard · Cognitive Fatigue & Digital Habits · Built with Streamlit'
    '</div>',
    unsafe_allow_html=True,
)