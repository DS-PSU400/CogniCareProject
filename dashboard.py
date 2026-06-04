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

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #F8F9FA;
    color: #1A1A2E;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #DEE2E6;
}

/* Title */
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

/* Metric cards */
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
.metric-delta {
    font-size: 0.78rem;
    color: #2E7D32;
    margin-top: 0.15rem;
}

/* Section headers */
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

/* Insight boxes */
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

/* Fatigue badge */
.badge-refreshed { background:#E8F5E9; color:#2E7D32; border:1px solid #A5D6A7; border-radius:20px; padding:3px 14px; font-size:0.78rem; }
.badge-strained   { background:#FFF8E1; color:#F57F17; border:1px solid #FFE082; border-radius:20px; padding:3px 14px; font-size:0.78rem; }
.badge-burnout    { background:#FFEBEE; color:#C62828; border:1px solid #EF9A9A; border-radius:20px; padding:3px 14px; font-size:0.78rem; }

/* Plotly chart background */
.js-plotly-plot .plotly { background: transparent !important; }

/* Sidebar filters */
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
PLOT_BG   = "#FFFFFF"
PAPER_BG  = "#F8F9FA"
GRID_COL  = "#E9ECEF"
TEXT_COL  = "#495057"
FONT_FAM  = "DM Sans"
PALETTE   = {"refreshed": "#2E7D32", "strained": "#F57F17", "near-burnout": "#C62828"}

def apply_chart_theme(fig, height=380):
    fig.update_layout(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_FAM, color=TEXT_COL, size=12),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        yaxis=dict(gridcolor=GRID_COL, zerolinecolor=GRID_COL),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=GRID_COL,
            font=dict(color=TEXT_COL),
        ),
    )
    return fig

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_cognitive_fatigue_dataset.csv")

    # Buat fatigue_level dari fatigue_score jika belum ada
    if "fatigue_level" not in df.columns and "fatigue_score" in df.columns:
        def fatigue_label(score):
            if score <= 2.0:
                return "refreshed"
            elif score <= 3.5:
                return "strained"
            else:
                return "near-burnout"
        df["fatigue_level"] = df["fatigue_score"].apply(fatigue_label)

    # Wellness index: ganti 0 dengan median
    if "wellness_index" in df.columns and (df["wellness_index"] == 0).any():
        df["wellness_index"] = df["wellness_index"].replace(0, np.nan)
        df["wellness_index"].fillna(df["wellness_index"].median(), inplace=True)

    # Hapus duplikat
    df.drop_duplicates(inplace=True)

    return df

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 CogniTrack")
    st.markdown("---")
    st.markdown('<div class="sidebar-label">Filter Data</div>', unsafe_allow_html=True)

    try:
        df_raw = load_data()
        data_loaded = True
    except FileNotFoundError:
        data_loaded = False

    if data_loaded:
        # Filter Mood
        if "mood" in df_raw.columns:
            moods_available = sorted(df_raw["mood"].dropna().unique().tolist())
            selected_moods = st.multiselect(
                "Mood",
                options=moods_available,
                default=moods_available,
            )
        else:
            selected_moods = None

        # Filter Fatigue Level
        fl_opts = ["refreshed", "strained", "near-burnout"]
        fl_opts_avail = [f for f in fl_opts if f in df_raw["fatigue_level"].unique()]
        selected_fl = st.multiselect(
            "Fatigue Level",
            options=fl_opts_avail,
            default=fl_opts_avail,
        )

        # Filter Screen Time
        st_min = float(df_raw["screen_time"].min())
        st_max = float(df_raw["screen_time"].max())
        st_range = st.slider(
            "Screen Time (jam)",
            min_value=st_min,
            max_value=st_max,
            value=(st_min, st_max),
            step=0.5,
        )

        # Terapkan filter
        df = df_raw.copy()
        if selected_moods and "mood" in df.columns:
            df = df[df["mood"].isin(selected_moods)]
        if selected_fl:
            df = df[df["fatigue_level"].isin(selected_fl)]
        df = df[(df["screen_time"] >= st_range[0]) & (df["screen_time"] <= st_range[1])]

        st.markdown("---")
        st.markdown(f"**{len(df):,}** baris tersaring dari **{len(df_raw):,}**")
    else:
        st.error("File `cleaned_cognitive_fatigue_dataset.csv` tidak ditemukan. Pastikan file ada di direktori yang sama.")
        df = None

    st.markdown("---")
    page = st.radio(
        "Navigasi",
        ["📊 Overview", "🔍 EDA & Distribusi", "🤝 Korelasi & RQ", "🧪 A/B Testing", "📋 Data Dictionary"],
    )

# ──────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 CogniCare</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Cognitive Fatigue & Digital Habits — Analytic Dashboard</div>', unsafe_allow_html=True)

if not data_loaded or df is None:
    st.warning("Upload file `cleaned_cognitive_fatigue_dataset.csv` ke direktori yang sama dengan `dashboard.py`.")
    st.stop()

# ═══════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════
if page == "📊 Overview":
    # ── KPI Row
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    def kpi_card(col, label, value, delta=""):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

    kpi_card(kpi1, "Total Responden",  f"{len(df):,}", "setelah filter")
    kpi_card(kpi2, "Avg Screen Time",  f"{df['screen_time'].mean():.1f}h", "per hari")
    kpi_card(kpi3, "Avg Sleep Hours",  f"{df['sleep_hours'].mean():.1f}h", "per hari")
    kpi_card(kpi4, "Avg Stress Level", f"{df['stress_level'].mean():.2f}", "skala 0–10")
    kpi_card(kpi5, "Avg Wellness",     f"{df['wellness_index'].mean():.1f}", "skala 0–100")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Fatigue Level Distribution + Mood Distribution
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Distribusi Fatigue Level</div>', unsafe_allow_html=True)
        fl_counts = df["fatigue_level"].value_counts().reset_index()
        fl_counts.columns = ["fatigue_level", "count"]
        fl_counts["pct"] = (fl_counts["count"] / fl_counts["count"].sum() * 100).round(1)

        fig_pie = px.pie(
            fl_counts,
            names="fatigue_level",
            values="count",
            color="fatigue_level",
            color_discrete_map=PALETTE,
            hole=0.55,
            custom_data=["pct"],
        )
        fig_pie.update_traces(
            texttemplate="%{label}<br><b>%{customdata[0]:.1f}%</b>",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>Jumlah: %{value:,}<br>Persen: %{customdata[0]:.1f}%<extra></extra>",
        )
        apply_chart_theme(fig_pie, height=340)
        fig_pie.add_annotation(
            text=f"<b>{len(df):,}</b><br>Responden",
            x=0.5, y=0.5, font_size=14,
            font_color="#1A1A2E", showarrow=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Distribusi Mood</div>', unsafe_allow_html=True)
        if "mood" in df.columns:
            mood_counts = df["mood"].value_counts().reset_index()
            mood_counts.columns = ["mood", "count"]
            fig_mood = px.bar(
                mood_counts,
                x="count", y="mood",
                orientation="h",
                color="count",
                color_continuous_scale=["#1565C0", "#00ACC1"],
                text="count",
            )
            fig_mood.update_traces(texttemplate="%{text:,}", textposition="outside")
            fig_mood.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Jumlah")
            apply_chart_theme(fig_mood, height=340)
            st.plotly_chart(fig_mood, use_container_width=True)
        else:
            st.info("Kolom 'mood' tidak tersedia.")

    # ── Wellness per Fatigue Level (box)
    st.markdown('<div class="section-header">Wellness Index per Fatigue Level</div>', unsafe_allow_html=True)
    fig_box = px.box(
        df, x="fatigue_level", y="wellness_index",
        color="fatigue_level",
        color_discrete_map=PALETTE,
        points="outliers",
        category_orders={"fatigue_level": ["refreshed", "strained", "near-burnout"]},
    )
    apply_chart_theme(fig_box, height=340)
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    💡 <b>Insight:</b> Pengguna dalam kategori <i>refreshed</i> konsisten menunjukkan wellness index lebih tinggi,
    sementara <i>near-burnout</i> memiliki sebaran yang lebih lebar — mengindikasikan faktor-faktor lain
    (screen time, sleep, aktivitas fisik) turut berkontribusi terhadap variasi kondisi kesehatan mental.
    </div>
    """, unsafe_allow_html=True)


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
        fig_hist = px.histogram(
            df, x=selected_num,
            color="fatigue_level",
            color_discrete_map=PALETTE,
            nbins=40, barmode="overlay", opacity=0.75,
            marginal="box",
            category_orders={"fatigue_level": ["refreshed", "strained", "near-burnout"]},
        )
        fig_hist.update_layout(title=f"Distribusi {selected_num} per Fatigue Level")
        apply_chart_theme(fig_hist, 380)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_vio = px.violin(
            df, y=selected_num, x="fatigue_level",
            color="fatigue_level",
            color_discrete_map=PALETTE,
            box=True, points=False,
            category_orders={"fatigue_level": ["refreshed", "strained", "near-burnout"]},
        )
        fig_vio.update_layout(title=f"Violin Plot: {selected_num}")
        apply_chart_theme(fig_vio, 380)
        st.plotly_chart(fig_vio, use_container_width=True)

    # Statistik Deskriptif
    st.markdown('<div class="section-header">Statistik Deskriptif</div>', unsafe_allow_html=True)
    stat = df[numeric_cols].describe().T.round(3)
    stat["skewness"] = df[numeric_cols].skew().round(3)
    stat["kurtosis"] = df[numeric_cols].kurtosis().round(3)
    st.dataframe(stat, use_container_width=True)

    # Deteksi Outlier
    st.markdown('<div class="section-header">Deteksi Outlier (Metode IQR)</div>', unsafe_allow_html=True)
    outlier_rows = []
    for col in numeric_cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = ((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum()
        outlier_rows.append({"Variabel": col, "Q1": round(q1,3), "Q3": round(q3,3),
                              "IQR": round(iqr,3), "Jumlah Outlier": n_out,
                              "Persen (%)": round(n_out/len(df)*100, 2)})
    st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)

    # Screen Time Distribution - time bins
    if "screen_time" in df.columns:
        st.markdown('<div class="section-header">Distribusi Screen Time (Bins)</div>', unsafe_allow_html=True)
        df_temp = df.copy()
        df_temp["screen_time_bins"] = pd.cut(
            df_temp["screen_time"],
            bins=[0,2,4,6,8,10,12,24],
            labels=["0-2","2-4","4-6","6-8","8-10","10-12","12+"]
        )
        bin_fl = df_temp.groupby(["screen_time_bins","fatigue_level"], observed=True).size().reset_index(name="count")
        fig_bin = px.bar(
            bin_fl, x="screen_time_bins", y="count",
            color="fatigue_level", color_discrete_map=PALETTE,
            barmode="stack",
            category_orders={"fatigue_level": ["refreshed","strained","near-burnout"]},
        )
        fig_bin.update_layout(xaxis_title="Screen Time (jam/hari)", yaxis_title="Jumlah Responden")
        apply_chart_theme(fig_bin, 360)
        st.plotly_chart(fig_bin, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        💡 <b>Insight:</b> Terlihat peningkatan proporsi <i>near-burnout</i> seiring screen time meningkat,
        terutama di atas 8 jam/hari. Ini mendukung rekomendasi threshold "Smart Alert" di angka 6–8 jam.
        </div>
        """, unsafe_allow_html=True)


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
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    apply_chart_theme(fig_heat, 420)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── RQ 1 – Scatter screen_time vs fatigue_score
    st.markdown('<div class="section-header">RQ1 · Screen Time vs Fatigue Score</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2,1])
    with col1:
        if "fatigue_score" in df.columns:
            fig_sc = px.scatter(
                df.sample(min(3000, len(df)), random_state=42),
                x="screen_time", y="fatigue_score",
                color="fatigue_level", color_discrete_map=PALETTE,
                opacity=0.5,
                trendline="ols",
                trendline_scope="overall",
                trendline_color_override="#1565C0",
                category_orders={"fatigue_level": ["refreshed","strained","near-burnout"]},
            )
            apply_chart_theme(fig_sc, 380)
            st.plotly_chart(fig_sc, use_container_width=True)
    with col2:
        st.markdown("""
        <div class="insight-box" style="margin-top:3rem">
        📌 <b>Temuan:</b> Terdapat korelasi positif antara screen time dan fatigue score.
        Semakin lama durasi layar, skor kelelahan kognitif cenderung meningkat,
        meskipun terdapat variasi yang dipengaruhi faktor lain (sleep, aktivitas fisik).
        </div>
        """, unsafe_allow_html=True)

    # ── RQ 2 – Sleep Hours vs Wellness Index
    st.markdown('<div class="section-header">RQ2 · Sleep Hours vs Wellness Index</div>', unsafe_allow_html=True)
    if "wellness_index" in df.columns:
        df_sleep = df.copy()
        df_sleep["sleep_bins"] = pd.cut(df_sleep["sleep_hours"],
                                        bins=[0,4,6,7,8,9,12],
                                        labels=["<4","4-6","6-7","7-8","8-9",">9"])
        sleep_agg = df_sleep.groupby("sleep_bins", observed=True)["wellness_index"].mean().reset_index()

        fig_sleep = px.line(
            sleep_agg, x="sleep_bins", y="wellness_index",
            markers=True,
            line_shape="spline",
        )
        fig_sleep.update_traces(line_color="#1565C0", marker_color="#00ACC1", marker_size=10)
        fig_sleep.update_layout(xaxis_title="Durasi Tidur (jam)", yaxis_title="Rata-rata Wellness Index")
        apply_chart_theme(fig_sleep, 340)
        st.plotly_chart(fig_sleep, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        💡 Wellness index tertinggi ditemukan pada rentang tidur 7–8 jam — konsisten dengan rekomendasi
        kesehatan umum. Di bawah 6 jam atau di atas 9 jam, wellness cenderung menurun.
        </div>
        """, unsafe_allow_html=True)

    # ── RQ 3 – Stress Level vs Wellness
    st.markdown('<div class="section-header">RQ3 · Stress Level vs Wellness Index</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        fig_str = px.scatter(
            df.sample(min(3000, len(df)), random_state=1),
            x="stress_level", y="wellness_index",
            color="fatigue_level", color_discrete_map=PALETTE,
            opacity=0.45,
            trendline="ols",
            trendline_scope="overall",
            trendline_color_override="#1565C0",
        )
        apply_chart_theme(fig_str, 360)
        st.plotly_chart(fig_str, use_container_width=True)

    with col4:
        # ── RQ 4 – Mood Comparison
        if "mood" in df.columns:
            profile_cols = [c for c in ["screen_time","sleep_hours","physical_activity"] if c in df.columns]
            profile = df.groupby("mood")[profile_cols].mean().reset_index()
            fig_prof = px.bar(
                profile.melt(id_vars="mood", value_vars=profile_cols),
                x="variable", y="value", color="mood",
                barmode="group",
            )
            fig_prof.update_layout(title="Profil Kebiasaan per Mood",
                                   xaxis_title="", yaxis_title="Rata-rata Nilai")
            apply_chart_theme(fig_prof, 360)
            st.plotly_chart(fig_prof, use_container_width=True)

    # ── RQ 5 – Physical Activity Mitigasi Screen Time
    st.markdown('<div class="section-header">RQ5 · Mitigasi Aktivitas Fisik terhadap Efek Screen Time</div>', unsafe_allow_html=True)
    if "physical_activity" in df.columns:
        df_rq5 = df.copy()
        df_rq5["Activity_Level"] = pd.cut(df_rq5["physical_activity"],
                                           bins=[-1,3,100],
                                           labels=["Low Activity","High Activity"])
        df_rq5["screen_time_bins"] = pd.cut(df_rq5["screen_time"],
                                             bins=[0,2,4,6,8,10,12,24],
                                             labels=["0-2","2-4","4-6","6-8","8-10","10-12","12+"])
        rq5_agg = (df_rq5.groupby(["screen_time_bins","Activity_Level"], observed=True)
                         ["wellness_index"].mean().reset_index())

        fig_rq5 = px.line(
            rq5_agg, x="screen_time_bins", y="wellness_index",
            color="Activity_Level",
            color_discrete_map={"Low Activity":"#f87171","High Activity":"#4ade80"},
            markers=True, line_shape="spline",
        )
        fig_rq5.update_layout(xaxis_title="Screen Time (jam)", yaxis_title="Rata-rata Wellness Index")
        apply_chart_theme(fig_rq5, 360)
        st.plotly_chart(fig_rq5, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        💡 <b>Temuan RQ5:</b> Kelompok "High Activity" mempertahankan wellness index yang lebih stabil
        meskipun screen time meningkat — membuktikan efek mitigasi aktivitas fisik dan menjadi dasar
        fitur "Digital Balance Meter".
        </div>
        """, unsafe_allow_html=True)

    # ── RQ 6 – Feature Importance (Korelasi terhadap Wellness)
    st.markdown('<div class="section-header">RQ6 · Ranking Fitur terhadap Wellness Index</div>', unsafe_allow_html=True)
    if "wellness_index" in df.columns:
        num_cols_rq6 = [c for c in numeric_cols if c != "wellness_index"]
        corr_target = df[num_cols_rq6 + ["wellness_index"]].corr()["wellness_index"].drop("wellness_index").sort_values()
        colors_rq6 = ["#f87171" if v < 0 else "#4ade80" for v in corr_target.values]

        fig_rq6 = go.Figure(go.Bar(
            x=corr_target.values,
            y=corr_target.index,
            orientation="h",
            marker_color=colors_rq6,
            text=[f"{v:.3f}" for v in corr_target.values],
            textposition="outside",
        ))
        fig_rq6.update_layout(xaxis_title="Koefisien Korelasi Pearson",
                               xaxis=dict(range=[-1,1], gridcolor=GRID_COL))
        apply_chart_theme(fig_rq6, 340)
        st.plotly_chart(fig_rq6, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 4: A/B TESTING
# ═══════════════════════════════════════════════
elif page == "🧪 A/B Testing":
    from scipy import stats

    st.markdown("""
    <div class="insight-box" style="border-left-color:#1565C0; margin-bottom:1.5rem">
    🔬 <b>Skenario A/B Testing:</b> Apakah terdapat perbedaan signifikan pada <i>Wellness Index</i>
    antara pengguna dengan screen time <b>rendah (≤ 6 jam)</b> vs screen time <b>tinggi (> 6 jam)</b>?
    </div>
    """, unsafe_allow_html=True)

    col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
    with col_cfg1:
        threshold = st.number_input("Threshold Screen Time (jam)", min_value=1.0, max_value=20.0,
                                    value=6.0, step=0.5)
    with col_cfg2:
        alpha = st.selectbox("Significance Level (α)", [0.01, 0.05, 0.10], index=1)
    with col_cfg3:
        metric_col = st.selectbox("Metrik yang diuji",
                                  [c for c in ["wellness_index","fatigue_score","stress_level","sleep_hours"]
                                   if c in df.columns])

    # Bagi grup
    group_a = df[df["screen_time"] <= threshold][metric_col].dropna()
    group_b = df[df["screen_time"] >  threshold][metric_col].dropna()

    # T-test independen
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    cohen_d = (group_a.mean() - group_b.mean()) / np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)

    # Hasil
    st.markdown('<div class="section-header">Hasil Uji Statistik (Welch\'s t-test)</div>', unsafe_allow_html=True)

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

    # Kesimpulan
    if p_value < alpha:
        st.success(f"✅ **H₀ DITOLAK** (p = {p_value:.4f} < α = {alpha})  "
                   f"→ Terdapat perbedaan yang **signifikan secara statistik** pada {metric_col} "
                   f"antara kedua kelompok.  Cohen's d = {cohen_d:.3f}")
    else:
        st.info(f"ℹ️ **H₀ GAGAL DITOLAK** (p = {p_value:.4f} ≥ α = {alpha})  "
                f"→ Tidak cukup bukti untuk menyimpulkan perbedaan signifikan. Cohen's d = {cohen_d:.3f}")

    # Visualisasi distribusi kedua grup
    st.markdown('<div class="section-header">Distribusi Kedua Grup</div>', unsafe_allow_html=True)
    fig_ab = go.Figure()
    fig_ab.add_trace(go.Histogram(
        x=group_a, name=f"Group A (≤{threshold}h)", opacity=0.7,
        marker_color="#1565C0", nbinsx=40,
    ))
    fig_ab.add_trace(go.Histogram(
        x=group_b, name=f"Group B (>{threshold}h)", opacity=0.7,
        marker_color="#f87171", nbinsx=40,
    ))
    fig_ab.add_vline(x=group_a.mean(), line_dash="dash", line_color="#1565C0",
                     annotation_text=f"Mean A={group_a.mean():.2f}", annotation_position="top right")
    fig_ab.add_vline(x=group_b.mean(), line_dash="dash", line_color="#f87171",
                     annotation_text=f"Mean B={group_b.mean():.2f}", annotation_position="top left")
    fig_ab.update_layout(barmode="overlay", xaxis_title=metric_col, yaxis_title="Frekuensi")
    apply_chart_theme(fig_ab, 380)
    st.plotly_chart(fig_ab, use_container_width=True)

    # Ringkasan statistik
    st.markdown('<div class="section-header">Ringkasan Statistik Deskriptif</div>', unsafe_allow_html=True)
    summary_ab = pd.DataFrame({
        "Statistik": ["N", "Mean", "Median", "Std Dev", "Min", "Max"],
        f"Group A (≤{threshold}h)": [
            len(group_a), group_a.mean().round(3), group_a.median().round(3),
            group_a.std().round(3), group_a.min().round(3), group_a.max().round(3)
        ],
        f"Group B (>{threshold}h)": [
            len(group_b), group_b.mean().round(3), group_b.median().round(3),
            group_b.std().round(3), group_b.min().round(3), group_b.max().round(3)
        ],
    })
    st.dataframe(summary_ab, use_container_width=True)

    # Effect size interpretation
    st.markdown('<div class="section-header">Interpretasi Effect Size (Cohen\'s d)</div>', unsafe_allow_html=True)
    abs_d = abs(cohen_d)
    if abs_d < 0.2:
        effect_label = "Negligible (sangat kecil)"
    elif abs_d < 0.5:
        effect_label = "Small (kecil)"
    elif abs_d < 0.8:
        effect_label = "Medium (sedang)"
    else:
        effect_label = "Large (besar)"

    st.markdown(f"""
    <div class="insight-box">
    📐 <b>Cohen's d = {cohen_d:.3f}</b> → <b>{effect_label}</b><br><br>
    Panduan: |d| &lt; 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, ≥ 0.8 = large.<br>
    Nilai p-value yang signifikan dengan effect size kecil perlu diinterpretasikan hati-hati —
    pada dataset besar, perbedaan kecil pun bisa signifikan secara statistik tetapi tidak bermakna secara praktis.
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# PAGE 5: DATA DICTIONARY
# ═══════════════════════════════════════════════
elif page == "📋 Data Dictionary":
    st.markdown('<div class="section-header">Data Dictionary</div>', unsafe_allow_html=True)

    data_dict = {
        "Variabel": ["screen_time", "sleep_hours", "stress_level", "wellness_index",
                     "physical_activity", "fatigue_score", "mood", "fatigue_level"],
        "Tipe Data": ["Float", "Float", "Float/Int", "Float", "Float/Int", "Float/Int", "Object", "Object"],
        "Satuan / Skala": ["Jam/hari", "Jam/hari", "0–10", "0–100", "Jam/hari", "0–10 (approx.)",
                           "Kategorik", "Kategorik (3 kelas)"],
        "Deskripsi": [
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
            "Smart Alert: > 6–8 jam",
            "Optimal: 7–8 jam",
            "Tinggi: > 7",
            "Rendah: < 40, Sedang: 40–70, Tinggi: > 70",
            "Low: ≤ 3 jam, High: > 3 jam",
            "≤ 2.0 → refreshed | 2.0–3.5 → strained | > 3.5 → near-burnout",
            "—",
            "Target variable (multi-class classification)",
        ],
    }

    df_dict = pd.DataFrame(data_dict)
    st.dataframe(df_dict, use_container_width=True, height=340)

    # Info Dataset
    st.markdown('<div class="section-header">Info Dataset</div>', unsafe_allow_html=True)
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("Jumlah Baris (raw)", "98,535")
    col_i2.metric("Jumlah Kolom", str(len(df.columns)))
    col_i3.metric("Missing Value", "0")
    col_i4.metric("Duplikat", "920 (dihapus)")

    # Preview data
    st.markdown('<div class="section-header">Preview Data (Tersaring)</div>', unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=300)

    st.markdown("""
    <div class="insight-box">
    📁 <b>Sumber Data:</b><br>
    • Dataset A: <code>digital_habits_vs_mental_health.csv</code> (utama, n ≈ 98.535)<br>
    • Dataset B: <code>ScreenTime vs MentalWellness.csv</code> (validasi silang)<br>
    • Target variable <code>fatigue_level</code> di-derive dari <code>fatigue_score</code>
      menggunakan threshold berbasis distribusi kuartil.
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; color:#3a4060; font-size:0.78rem; font-family:Space Mono,monospace;">'
    'CogniCare Dashboard · Cognitive Fatigue & Digital Habits · Built with Streamlit'
    '</div>',
    unsafe_allow_html=True
)
