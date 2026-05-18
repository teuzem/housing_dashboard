"""
Housing Price Analysis — Tableau de Bord Décisionnel
Application Streamlit — Analyse exploratoire complète
Développée par NGOUMTSOP TEUZEM Yeiayel Chavaquiah
Master 1 Data Science — ISJ
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# CONFIGURATION DE LA PAGE
st.set_page_config(
    page_title="Dashboard Analytique des prix de biens immobiliers",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

TABLES_DIR = "tables"
os.makedirs(TABLES_DIR, exist_ok=True)

# styles CSS
st.markdown("""
<style>
    .block-container { padding: 1.5rem 2rem 4.5rem 2rem; }
    h1  { font-size: 1.65rem !important; font-weight: 700; color: #1a2b4a; }
    h2  { font-size: 1.15rem !important; font-weight: 600; color: #2c3e50;
          border-bottom: 2px solid #e8ecf0; padding-bottom: 0.4rem; margin-top: 1.5rem; }
    h3  { font-size: 0.95rem !important; color: #4a5568; }

    /* Metric card */
    .metric-card {
        background:#fff; border:1px solid #e2e8f0; border-radius:8px;
        padding:1rem 1.25rem; text-align:center;
        box-shadow:0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-card .label { font-size:.75rem; color:#718096; text-transform:uppercase;
                          letter-spacing:.04em; margin-bottom:.25rem; }
    .metric-card .value { font-size:1.6rem; font-weight:700; color:#1a2b4a; }
    .metric-card .sub   { font-size:.75rem; color:#a0aec0; margin-top:.15rem; }

    /* Insight badges — display:block assures full-width rendering */
    .insight-badge {
        display: block !important;
        background: #eef4fb;
        border-left: 4px solid #2980b9;
        border-radius: 4px;
        padding: .55rem .85rem;
        font-size: .83rem;
        color: #2c3e50;
        margin-bottom: .5rem;
        width: 100%;
        box-sizing: border-box;
        line-height: 1.5;
    }
    .insight-badge.green  { background:#edfaf1; border-color:#27ae60; color:#1a5631; }
    .insight-badge.red    { background:#fff3e0; border-color:#e67e22; color:#7d3c05; }
    .insight-badge.orange { background:#fef9e7; border-color:#f1c40f; color:#7d6608; }
    .insight-badge.purple { background:#f5eef8; border-color:#8e44ad; color:#4a235a; }

    /* Warning box */
    .warning-box {
        background:#fff3cd; border:1px solid #ffc107; border-radius:6px;
        padding:.6rem 1rem; font-size:.83rem; color:#856404; margin-bottom:.75rem;
    }

    /* Tables */
    .styled-table { width:100%; border-collapse:collapse; font-size:.82rem; }
    .styled-table th { background:#f7f9fb; color:#4a5568; font-weight:600;
                       padding:.5rem .75rem; text-align:left; border-bottom:2px solid #e2e8f0; }
    .styled-table td { padding:.45rem .75rem; border-bottom:1px solid #f0f4f8; color:#2d3748; }
    .styled-table tr:hover td { background:#f7faff; }

    /* Sidebar */
    [data-testid="stSidebar"] { background:#1a2b4a; }
    [data-testid="stSidebar"] * { color:#cbd5e0 !important; }
    [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3
        { color:#e2e8f0 !important; border-bottom-color:#2d4a6e !important; }

    /* Section divider */
    .section-divider { border:none; border-top:1px solid #e2e8f0; margin:1.5rem 0; }

    /* Footer*/
    .footer {
        position:fixed; bottom:0; left:0; right:0; z-index:9999;
        background:linear-gradient(90deg,#1a2b4a 0%,#2c4a7c 100%);
        color:#a0b8d0; text-align:center;
        padding:.42rem 1rem; font-size:.72rem;
        border-top:1px solid #2d4a6e; letter-spacing:.02em;
        align-items: center;
        justify-content: center;
        margin-left: 290px;
    }
    .footer strong { color:#e2e8f0; }
</style>
""", unsafe_allow_html=True)

# MATPLOTLIB THEME 
plt.rcParams.update({
    "figure.dpi": 120, "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA", "axes.grid": True,
    "grid.alpha": .35, "grid.linestyle": "--",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
})

BLUE   = "#2980B9"
NAVY   = "#1a2b4a"
GREEN  = "#27AE60"
RED    = "#E74C3C"
ORANGE = "#E67E22"
PURPLE = "#8E44AD"
SEQ    = "Blues"
DIV    = "RdBu_r"
CAT_PAL = [BLUE, GREEN, RED, PURPLE, ORANGE, "#16A085"]

# DATA LOADING
@st.cache_data
def load_data():
    df = pd.read_csv("Housing.csv")
    for col in ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]:
        df[col] = df[col].map({"yes": 1, "no": 0})
    df["price_M"]        = df["price"] / 1_000_000
    df["price_per_sqft"] = df["price"] / df["area"]
    df["price_segment"]  = pd.qcut(df["price"], q=3, labels=["Bas","Moyen","Élevé"])
    return df

df = load_data()

binary_cols = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","prefarea"]
num_cols    = ["price","area","bedrooms","bathrooms","stories","parking"]
# Uniquement variables quantitatives (discrètes ou continues) 
ord_cols    = ["bedrooms","bathrooms","stories","parking"]

EQUIP_LABELS = {
    "mainroad":       "Route principale",
    "guestroom":      "Chambre d'invités",
    "basement":       "Cave / sous-sol",
    "hotwaterheating":"Eau chaude",
    "airconditioning":"Climatisation",
    "prefarea":       "Zone préférentielle",
}

# HELPERS 
def metric_card(label, value, sub=""):
    return f"""<div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>"""

def insight_html(text, kind="blue"):
    cls = ("insight-badge green"  if kind == "green"
           else "insight-badge red"    if kind == "red"
           else "insight-badge orange" if kind == "orange"
           else "insight-badge purple" if kind == "purple"
           else "insight-badge")
    return f'<div class="{cls}">{text}</div>'

def show_insight(text, kind="blue"):
    """Affiche un badge insight en rendant correctement le HTML."""
    st.markdown(insight_html(text, kind), unsafe_allow_html=True)

def footer():
    st.markdown("""
    <div class="footer">
        © 2026 Tous droits reservés &nbsp;|&nbsp;
        Développé par <strong>NGOUMTSOP TEUZEM Yeiayel Chavaquiah</strong> &nbsp;—&nbsp;
        Étudiant en <strong>Master 1 Data Science</strong> à l'<strong>ISJ</strong>
        &nbsp;|&nbsp; Dashboard Analytique des prix de biens immobiliers
    </div>
    """, unsafe_allow_html=True)

def save_table(df_t, filename):
    df_t.to_csv(os.path.join(TABLES_DIR, filename))

def to_csv_bytes(df_t):
    return df_t.to_csv(index=True).encode("utf-8")

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='text-align:left;padding:1rem 0 0.8rem 0;'>
        <div style='font-size:1.4rem;font-weight:700;color:#e2e8f0;letter-spacing:.03em;'>
            🏠 App Tableau de bord Analytique des prix des biens immobiliers EDA.
        </div>
        <div style='font-size:0.8rem;color:#7b93b0;margin-top:.2rem;'>
            Un tableau de bord analytique des prix de biens immobiliers, un outil interactif qui centralise, synthétise et analyse les données "Housing.csv" et transforme les données complexes en indicateurs clés de performance (KPI) pour faciliter la prise de décision des investisseurs.
        </div>
    </div>
    <hr style='border-color:#2d4a6e;margin:.75rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("#### Navigation")
    page = st.radio("", [
        "Vue d'ensemble",
        "Analyse univariée",
        "Analyse bivariée",
        "Analyse multivariée",
        "Insights décisionnels",
        "Exports globaux",
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#2d4a6e;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("#### Filtres globaux")

    price_range = st.slider(
        "Plage de prix (millions)",
        float(df["price_M"].min()), float(df["price_M"].max()),
        (float(df["price_M"].min()), float(df["price_M"].max())),
        step=0.1, format="%.1f M"
    )
    furn_opts = ["Tous"] + sorted(df["furnishingstatus"].unique().tolist())
    furn_sel  = st.selectbox("Ameublement", furn_opts)
    area_range = st.slider(
        "Superficie (sq ft)",
        int(df["area"].min()), int(df["area"].max()),
        (int(df["area"].min()), int(df["area"].max())), step=50
    )

    st.markdown("<hr style='border-color:#2d4a6e;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.7rem;color:#4a6080;text-align:center;line-height:1.6;'>
        Dataset : 545 propriétés<br>
        13 variables · 0 valeur manquante<br>
        Source : Kaggle Housing Dataset
    </div>
    """, unsafe_allow_html=True)

# FILTRES GENERAUX
mask = (
    (df["price_M"] >= price_range[0]) & (df["price_M"] <= price_range[1]) &
    (df["area"]    >= area_range[0])  & (df["area"]    <= area_range[1])
)
if furn_sel != "Tous":
    mask &= df["furnishingstatus"] == furn_sel
dff = df[mask].copy()
n_filtered = len(dff)


# PAGE : VUE D'ENSEMBLE
if page == "Vue d'ensemble":
    st.markdown("## Vue d'ensemble du marché immobilier des biens")
    st.caption(f"Données filtrées : **{n_filtered}** propriétés sur 545")

    cv_prix = dff["price"].std() / dff["price"].mean() * 100
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.markdown(metric_card("Propriétés",    f"{n_filtered:,}",                 "dans la sélection"),         unsafe_allow_html=True)
    c2.markdown(metric_card("Prix médian",   f"{dff['price'].median()/1e6:.3f} M", "unité monétaire"),        unsafe_allow_html=True)
    c3.markdown(metric_card("Prix moyen",    f"{dff['price'].mean()/1e6:.3f} M",   f"σ ± {dff['price'].std()/1e6:.3f} M"), unsafe_allow_html=True)
    c4.markdown(metric_card("CV du prix",    f"{cv_prix:.1f}%",                 "coef de variation"),  unsafe_allow_html=True)
    c5.markdown(metric_card("Superficie moy.",f"{dff['area'].mean():.0f} ft²",  "surface habitable"),         unsafe_allow_html=True)
    c6.markdown(metric_card("Prix/ft² moy.", f"{dff['price_per_sqft'].mean():.0f}", "unité par ft²"),         unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("### Distribution du prix")
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        axes[0].hist(dff["price_M"], bins=28, color=BLUE, edgecolor="white", linewidth=.4, alpha=.85)
        axes[0].axvline(dff["price"].median()/1e6, color=RED,    lw=1.5, ls="--", label="Médiane")
        axes[0].axvline(dff["price"].mean()/1e6,   color=ORANGE, lw=1.5, ls="-",  label="Moyenne")
        axes[0].set_xlabel("Prix (millions)"); axes[0].set_ylabel("Nb propriétés")
        axes[0].set_title("Histogramme du prix"); axes[0].legend(fontsize=8)

        sk = dff["price"].skew(); ku = dff["price"].kurtosis()
        axes[1].hist(np.log(dff["price"]), bins=28, color=GREEN, edgecolor="white", lw=.4, alpha=.85)
        axes[1].set_xlabel("log(Prix)"); axes[1].set_ylabel("Fréquence")
        axes[1].set_title("Distribution log-transformée")
        axes[1].text(.97,.95, f"Skewness : {sk:.2f}\nKurtosis : {ku:.2f}",
                     transform=axes[1].transAxes, ha="right", va="top", fontsize=8,
                     bbox=dict(boxstyle="round,pad=.3", fc="white", ec="#e2e8f0"))
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    with col_b:
        st.markdown("### Répartition par segment")
        seg_counts = dff["price_segment"].value_counts().reindex(["Bas","Moyen","Élevé"])
        fig, ax = plt.subplots(figsize=(5, 3.8))
        wedges, _, autotexts = ax.pie(
            seg_counts.values, labels=seg_counts.index,
            colors=[BLUE, GREEN, NAVY], autopct="%1.1f%%",
            startangle=90, pctdistance=.78,
            wedgeprops=dict(linewidth=1.5, edgecolor="white"))
        for at in autotexts: at.set_fontsize(9)
        ax.set_title("Segments de prix (terciles)")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    show_insight("Dans toute cette étude tous les prix des propriétés sont en Millions de FCFA")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("### Statistiques descriptives")
        desc = dff[num_cols].describe().T[["mean","50%","std","min","max"]].round(4)
        desc.columns = ["Moyenne","Médiane","Écart-type","Min","Max"]
        desc.index = ["Prix (M)" if i=="price" else i for i in desc.index]
        for stat in ["Moyenne","Médiane","Écart-type","Min","Max"]:
            desc.loc["Prix (M)", stat] = round(float(desc.loc["Prix (M)", stat]) / 1e6, 4)
        desc["CV (%)"] = (desc["Écart-type"] / desc["Moyenne"] * 100).round(2)
        st.dataframe(desc, use_container_width=True)

    with col_d:
        st.markdown("### Équipements présents")
        rows_eq = []
        for col in binary_cols:
            pct = dff[col].mean() * 100
            prix_avec = dff[dff[col]==1]["price"].mean()/1e6 if dff[col].sum()>0 else None
            rows_eq.append({
                "Équipement": EQUIP_LABELS[col],
                "Présent (%)": f"{pct:.1f}%",
                "Nb propriétés": int(dff[col].sum()),
                "Prix moy. avec (M)": round(prix_avec, 3) if prix_avec else "—",
            })
        eq_df = pd.DataFrame(rows_eq)
        st.dataframe(eq_df, use_container_width=True, hide_index=True)
        save_table(eq_df, "02_equipements_overview.csv")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Prix au sq ft par segment")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(dff["price_per_sqft"], bins=30, color=PURPLE, edgecolor="white", alpha=.85)
    axes[0].axvline(dff["price_per_sqft"].median(), color=RED,    ls="--", lw=1.5, label="Médiane")
    axes[0].axvline(dff["price_per_sqft"].mean(),   color=ORANGE, lw=1.5, label="Moyenne")
    axes[0].set_xlabel("Prix / sq ft"); axes[0].set_ylabel("Fréquence")
    axes[0].set_title("Distribution du Prix au sq ft"); axes[0].legend(fontsize=8)

    seg_ppsf = dff.groupby("price_segment", observed=False)["price_per_sqft"].mean().reindex(["Bas","Moyen","Élevé"])
    bars_s = axes[1].bar(seg_ppsf.index, seg_ppsf.values, color=[BLUE,GREEN,NAVY], edgecolor="white")
    axes[1].set_ylabel("Prix moyen / sq ft"); axes[1].set_title("Prix/sq ft moyen par segment")
    for bar in bars_s:
        axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                     f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)
    footer()


# PAGE : ANALYSE UNIVARIÉE
elif page == "Analyse univariée":
    st.markdown("## Analyse univariée")
    st.caption("Distribution individuelle de chaque variable")

    tab1, tab2, tab3 = st.tabs(["Variables quantitatives", "Variables catégorielles", "Tests de normalité"])

    # Tab 1 : quantitatives
    with tab1:
        var = st.selectbox("Choisir une variable", num_cols)
        data = dff[var] / 1e6 if var == "price" else dff[var]
        label = f"{var} (millions)" if var == "price" else var

        m1,m2,m3,m4,m5,m6 = st.columns(6)
        m1.metric("Moyenne",    f"{data.mean():.4f}")
        m2.metric("Médiane",    f"{data.median():.4f}")
        m3.metric("Écart-type", f"{data.std():.4f}")
        m4.metric("Skewness",   f"{data.skew():.4f}")
        m5.metric("Kurtosis",   f"{data.kurtosis():.4f}")
        m6.metric("CV (%)",     f"{data.std()/data.mean()*100:.2f}%")

        if dff[var].nunique() <= 7:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            counts_v = dff[var].value_counts().sort_index()
            axes[0].bar(counts_v.index.astype(str), counts_v.values, color=BLUE, edgecolor="white")
            for bar in axes[0].patches:
                axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.3,
                             f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8)
            axes[0].set_title(f"Fréquences — {var}"); axes[0].set_xlabel(var); axes[0].set_ylabel("Nombre")
            axes[1].boxplot(data, patch_artist=True,
                            boxprops=dict(facecolor=BLUE, alpha=.5),
                            medianprops=dict(color=RED, linewidth=2),
                            flierprops=dict(marker="o", color=RED, markersize=4, alpha=.5))
            axes[1].set_title(f"Boxplot — {var}"); axes[1].set_ylabel(label)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].hist(data, bins=28, color=BLUE, edgecolor="white", lw=.4)
            axes[0].set_title(f"Histogramme — {var}"); axes[0].set_xlabel(label); axes[0].set_ylabel("Fréquence")
            stats.probplot(dff[var], dist="norm", plot=axes[1])
            axes[1].set_title("Q-Q Plot (normalité)")
            axes[1].get_lines()[0].set(color=BLUE, markersize=3)
            axes[1].get_lines()[1].set(color=RED, linewidth=1.5)
            axes[2].boxplot(data, patch_artist=True,
                            boxprops=dict(facecolor=BLUE, alpha=.5),
                            medianprops=dict(color=RED, linewidth=2),
                            flierprops=dict(marker="o", color=RED, markersize=4, alpha=.5))
            axes[2].set_title(f"Boxplot — {var}"); axes[2].set_ylabel(label)

        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        q1_v, q3_v = dff[var].quantile([.25, .75])
        iqr_v = q3_v - q1_v
        n_out = ((dff[var] < q1_v - 1.5*iqr_v) | (dff[var] > q3_v + 1.5*iqr_v)).sum()

        st.markdown("##### Statistiques avancées")
        a1,a2,a3,a4,a5,a6 = st.columns(6)
        a1.metric("Q1 (25%)", f"{data.quantile(.25):.4f}")
        a2.metric("Q3 (75%)", f"{data.quantile(.75):.4f}")
        a3.metric("IQR", f"{data.quantile(.75)-data.quantile(.25):.4f}")
        a4.metric("Outliers (IQR)", str(n_out))
        a5.metric("% Outliers", f"{n_out/len(data)*100:.1f}%")
        a6.metric("Étendue", f"{data.max()-data.min():.4f}")

        st.markdown("##### Tableau des percentiles")
        pct_list = [1,5,10,25,50,75,90,95,99]
        pct_df = pd.DataFrame({
            "Percentile (%)": pct_list,
            "Valeur": [round(data.quantile(p/100), 5) for p in pct_list]
        })
        st.dataframe(pct_df, use_container_width=True, hide_index=True)

    # Tab 2 : catégorielles
    with tab2:
        st.markdown("### Répartition des variables catégorielles et binaires")
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle("Distribution des variables catégorielles", fontweight="bold")
        axes = axes.flatten()
        for i, col in enumerate(["furnishingstatus"] + binary_cols):
            ax = axes[i]
            if col == "furnishingstatus":
                counts_c = dff[col].value_counts()
                ax.pie(counts_c.values, labels=counts_c.index, autopct="%1.1f%%",
                       colors=CAT_PAL[:len(counts_c)], startangle=90,
                       wedgeprops=dict(linewidth=1.2, edgecolor="white"))
            else:
                counts_c = dff[col].value_counts().sort_index()
                ax.bar(["Non","Oui"], counts_c.values, color=[BLUE,GREEN], edgecolor="white")
                ax.set_ylabel("Nombre")
                for bar, v in zip(ax.patches, counts_c.values):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                            f"{v/len(dff)*100:.1f}%", ha="center", va="bottom", fontsize=8)
            ax.set_title(EQUIP_LABELS.get(col, col)); ax.set_facecolor("#FAFAFA")
        axes[-1].set_visible(False)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown("##### Tableau de fréquence — Ameublement")
        furn_freq = dff["furnishingstatus"].value_counts().reset_index()
        furn_freq.columns = ["Statut","Fréquence"]
        furn_freq["Proportion (%)"] = (furn_freq["Fréquence"] / len(dff) * 100).round(1)
        st.dataframe(furn_freq, use_container_width=True, hide_index=True)

        st.markdown("##### Tableau de fréquence — Variables binaires")
        bin_rows = []
        for col in binary_cols:
            cnt1 = int(dff[col].sum()); cnt0 = len(dff) - cnt1
            bin_rows.append({
                "Variable": EQUIP_LABELS[col],
                "Non (nb)": cnt0, "Non (%)": f"{cnt0/len(dff)*100:.1f}%",
                "Oui (nb)": cnt1, "Oui (%)": f"{cnt1/len(dff)*100:.1f}%",
            })
        bin_freq_df = pd.DataFrame(bin_rows)
        st.dataframe(bin_freq_df, use_container_width=True, hide_index=True)

    # Tab 3 : Tests de normalité
    with tab3:
        st.markdown("### Tests de normalité formels")
        st.markdown("""<div class="warning-box">
            ℹ️ H₀ : la variable suit une distribution normale. 
            p-value &lt; 0.05 → on rejette H₀ (non-normalité détectée).
        </div>""", unsafe_allow_html=True)

        norm_rows = []
        for col in ["price","area"]:
            d_col = dff[col]
            sw_stat, sw_p = stats.shapiro(d_col.sample(min(len(d_col), 5000), random_state=42))
            jb_stat, jb_p = stats.jarque_bera(d_col)
            ad_res = stats.anderson(d_col, dist="norm")
            norm_rows.append({
                "Variable":           col,
                "Skewness":           round(d_col.skew(), 4),
                "Kurtosis":           round(d_col.kurtosis(), 4),
                "Shapiro-Wilk stat":  round(sw_stat, 5),
                "Shapiro-Wilk p":     round(sw_p, 6),
                "Jarque-Bera stat":   round(jb_stat, 2),
                "Jarque-Bera p":      round(jb_p, 6),
                "Anderson-Darling":   round(ad_res.statistic, 4),
                "Seuil critique 5%":  round(ad_res.critical_values[2], 4),
                "Normalité rejetée":  "Oui ✗" if sw_p < 0.05 or jb_p < 0.05 else "Non ✓",
            })
        norm_df = pd.DataFrame(norm_rows)
        st.dataframe(norm_df, use_container_width=True, hide_index=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for i, (col, ax) in enumerate(zip(["price","area"], axes)):
            stats.probplot(dff[col], dist="norm", plot=ax)
            ax.set_title(f"Q-Q Plot — {col}")
            ax.get_lines()[0].set(color=BLUE, markersize=3)
            ax.get_lines()[1].set(color=RED, linewidth=1.5)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        show_insight("Le prix et la superficie ne suivent pas une distribution normale (Shapiro-Wilk p ≈ 0). Des tests non-paramétriques (Mann-Whitney, Kruskal-Wallis) sont préférables.", "red")
        show_insight("La log-transformation du prix réduit l'asymétrie et rapproche la distribution de la normalité — recommandée pour la régression linéaire.", "orange")
        show_insight("Les variables discrètes (chambres, étages…) sont intrinsèquement non-normales : utiliser leur distribution empirique directement.", "orange")

    footer()

# PAGE : ANALYSE BIVARIÉE
elif page == "Analyse bivariée":
    st.markdown("## Analyse bivariée")
    st.caption("Relations entre les variables et le prix")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Prix vs Superficie", "Prix par catégorie",
        "Corrélations (quantitatives)", "Tests statistiques"
    ])

    # Tab 1
    with tab1:
        col_l, col_r = st.columns([3, 2])
        with col_l:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            axes[0].scatter(dff["area"], dff["price_M"], alpha=.35, s=18, color=BLUE, edgecolors="none")
            m, b, r, p, se = stats.linregress(dff["area"], dff["price"])
            x_line = np.linspace(dff["area"].min(), dff["area"].max(), 200)
            axes[0].plot(x_line, (m*x_line+b)/1e6, color=RED, lw=2, label=f"R²={r**2:.3f}")
            axes[0].set_xlabel("Superficie (sq ft)"); axes[0].set_ylabel("Prix (millions)")
            axes[0].set_title("Prix vs Superficie"); axes[0].legend(fontsize=8)

            color_var = st.selectbox("Colorer par", ["airconditioning","prefarea","furnishingstatus","aucun"])
            if color_var != "aucun":
                for j, cat in enumerate(sorted(dff[color_var].unique())):
                    sub = dff[dff[color_var]==cat]
                    lbl = str(cat) if color_var=="furnishingstatus" else ("Oui" if cat==1 else "Non")
                    axes[1].scatter(sub["area"], sub["price_M"], alpha=.4, s=18,
                                    color=CAT_PAL[j%len(CAT_PAL)], label=lbl, edgecolors="none")
                axes[1].legend(title=color_var, fontsize=7)
            else:
                axes[1].scatter(dff["area"], dff["price_M"], alpha=.35, s=18, color=BLUE, edgecolors="none")
            axes[1].set_xlabel("Superficie (sq ft)"); axes[1].set_ylabel("Prix (millions)")
            axes[1].set_title("Vue colorée")
            fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        with col_r:
            r2 = r**2
            show_insight(f"R² = {r2:.4f} — La superficie explique {r2*100:.1f}% de la variance du prix.")
            show_insight(f"Pente : +{m:.1f} unité de prix par sq ft supplémentaire.", "green")
            show_insight(f"p-value = {p:.2e} — corrélation hautement significative.", "green")
            show_insight(f"Erreur standard du coefficient : {se:.2f}", "orange")
            y_pred = m*dff["area"]+b
            res = dff["price"] - y_pred
            show_insight(f"Std des résidus : {res.std():.0f} — dispersion non expliquée par la superficie seule.", "orange")

        # Régression log
        st.markdown("##### Régression sur log(prix)")
        m_l, b_l, r_l, p_l, _ = stats.linregress(dff["area"], np.log(dff["price"]))
        fig2, ax2 = plt.subplots(figsize=(10, 3.5))
        ax2.scatter(dff["area"], np.log(dff["price"]), alpha=.3, s=14, color=GREEN, edgecolors="none")
        ax2.plot(x_line, m_l*x_line+b_l, color=RED, lw=2, label=f"R²={r_l**2:.3f}")
        ax2.set_xlabel("Superficie (sq ft)"); ax2.set_ylabel("log(Prix)")
        ax2.set_title("log(Prix) vs Superficie"); ax2.legend(fontsize=8)
        fig2.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close(fig2)
        show_insight(f"Modèle log : R² = {r_l**2:.4f} — légèrement supérieur au modèle linéaire, confirme la relation log-linéaire.", "green")

    # Tab 2 
    with tab2:
        var_cat = st.selectbox("Variable catégorielle", ord_cols + ["furnishingstatus"] + binary_cols)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        if dff[var_cat].nunique() <= 8:
            vals_c = sorted(dff[var_cat].unique())
            groups_c = [dff[dff[var_cat]==v]["price_M"].values for v in vals_c]
            lbl_c = [("Non" if v==0 else "Oui") if var_cat in binary_cols else str(v) for v in vals_c]
            bp = axes[0].boxplot(groups_c, labels=lbl_c, patch_artist=True,
                                 medianprops=dict(color=RED, linewidth=2))
            for j, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(CAT_PAL[j%len(CAT_PAL)]); patch.set_alpha(.6)
            axes[0].set_title(f"Distribution prix par {var_cat}"); axes[0].set_ylabel("Prix (millions)")

        grp_c = dff.groupby(var_cat)["price_M"].agg(["mean","median","count"]).reset_index()
        grp_c = grp_c.sort_values("mean", ascending=False)
        x_c = range(len(grp_c))
        bars_c = axes[1].bar(x_c, grp_c["mean"], color=BLUE, alpha=.8, edgecolor="white", label="Moyenne")
        axes[1].plot(x_c, grp_c["median"], "D", color=RED, markersize=7, label="Médiane", zorder=5)
        axes[1].set_xticks(x_c)
        lbl_x = [("Non" if str(v)=="0" else "Oui") if var_cat in binary_cols else str(v) for v in grp_c[var_cat]]
        axes[1].set_xticklabels(lbl_x, rotation=20, ha="right")
        axes[1].set_ylabel("Prix moyen (millions)"); axes[1].set_title(f"Prix moyen/médian par {var_cat}")
        axes[1].legend(fontsize=8)
        for bar in bars_c:
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02,
                         f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Stats par groupe
        grp_stats = dff.groupby(var_cat)["price_M"].agg(
            Nb="count", Moyenne="mean", Médiane="median", Std="std", Min="min", Max="max"
        ).round(3)
        st.markdown("##### Statistiques par groupe")
        st.dataframe(grp_stats, use_container_width=True)

        if var_cat in binary_cols:
            g0_b, g1_b = dff[dff[var_cat]==0]["price"], dff[dff[var_cat]==1]["price"]
            if len(g0_b) > 0 and len(g1_b) > 0:
                stat_mw, p_mw = stats.mannwhitneyu(g0_b, g1_b, alternative="two-sided")
                diff_b = (g1_b.mean()-g0_b.mean())/g0_b.mean()*100
                show_insight(
                    f"Test Mann-Whitney : p = {p_mw:.4f} — "
                    f"{'Différence significative' if p_mw<0.05 else '❌ Non significatif'} "
                    f"| Différence de prix : {diff_b:+.1f}%",
                    "green" if p_mw<0.05 else "red"
                )
        elif var_cat in ord_cols:
            groups_k = [dff[dff[var_cat]==v]["price"].values for v in sorted(dff[var_cat].unique()) if len(dff[dff[var_cat]==v])>0]
            if len(groups_k) >= 2:
                stat_k, p_k = stats.kruskal(*groups_k)
                show_insight(
                    f"Test Kruskal-Wallis : p = {p_k:.4f} — "
                    f"{'Différences significatives entre les groupes' if p_k<0.05 else '❌ Pas de différence significative'}",
                    "green" if p_k<0.05 else "red"
                )

    # Tab 3 : Corrélations quantitatives
    with tab3:
        st.markdown("### Matrice de corrélation — Variables quantitatives uniquement")
        st.markdown("""<div class="warning-box">
            ℹ️ La matrice est calculée <strong>exclusivement</strong> sur les variables
            quantitatives discrètes ou continues : prix, superficie, chambres, salles de bain,
            étages, parking. Les variables binaires (0/1) sont exclues : leur corrélation de
            Pearson n'est pas interprétable de la même manière que pour des variables continues.
        </div>""", unsafe_allow_html=True)

        corr_q = dff[num_cols].corr()
        mask_t = np.triu(np.ones_like(corr_q, dtype=bool))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        sns.heatmap(corr_q, mask=mask_t, annot=True, fmt=".2f", cmap=DIV,
                    center=0, vmin=-1, vmax=1, ax=axes[0],
                    linewidths=.5, linecolor="white", annot_kws={"size": 9})
        axes[0].set_title("Corrélations — Variables quantitatives (Pearson)")

        corr_price_q = corr_q["price"].drop("price").sort_values()
        colors_cp = [RED if v<0 else BLUE for v in corr_price_q.values]
        axes[1].barh(corr_price_q.index, corr_price_q.values, color=colors_cp, edgecolor="white")
        axes[1].axvline(0, color="black", lw=.8)
        for i, (idx, v) in enumerate(corr_price_q.items()):
            axes[1].text(v+(0.01 if v>=0 else -0.01), i, f"{v:.3f}",
                         va="center", ha="left" if v>=0 else "right", fontsize=8)
        axes[1].set_title("Corrélation avec le Prix"); axes[1].set_xlabel("Corrélation de Pearson")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown("##### Table des corrélations avec le prix")
        corr_with = corr_q["price"].drop("price").abs().sort_values(ascending=False).reset_index()
        corr_with.columns = ["Variable","| r | (avec Prix)"]
        corr_with["Direction"] = [("Evolution" if corr_q["price"][v]>0 else "−") for v in corr_with["Variable"]]
        corr_with["Corrélation r"] = [round(corr_q["price"][v],4) for v in corr_with["Variable"]]
        corr_with["Interprétation"] = corr_with["| r | (avec Prix)"].apply(
            lambda v: "Forte" if v>.5 else ("Modérée" if v>.3 else "Faible"))
        st.dataframe(corr_with, use_container_width=True, hide_index=True)

        show_insight("La superficie (area) présente la corrélation positive la plus élevée avec le prix parmi les variables quantitatives.", "green")
        show_insight("Le parking a la corrélation la plus faible — cohérent avec l'analyse bivariée et les tests statistiques.", "orange")

    #  Tab 4 : Tests statistiques
    with tab4:
        st.markdown("### Tableau synthétique des tests statistiques")
        test_rows = []
        for col in binary_cols:
            g0_t = dff[dff[col]==0]["price"]; g1_t = dff[dff[col]==1]["price"]
            if len(g0_t)==0 or len(g1_t)==0: continue
            stat_mw, p_mw = stats.mannwhitneyu(g0_t, g1_t, alternative="two-sided")
            diff_t = (g1_t.mean()-g0_t.mean())/g0_t.mean()*100
            eff_r = 1 - 2*stat_mw/(len(g0_t)*len(g1_t))
            test_rows.append({
                "Variable":         EQUIP_LABELS[col],
                "Prix sans (M)":    round(g0_t.mean()/1e6, 3),
                "Prix avec (M)":    round(g1_t.mean()/1e6, 3),
                "Différence (%)":   round(diff_t, 1),
                "Test":             "Mann-Whitney",
                "Statistique U":    round(stat_mw, 0),
                "p-value":          round(p_mw, 5),
                "Significatif":     "Oui" if p_mw<0.05 else "Non",
                "Taille effet r":   round(eff_r, 3),
            })
        for col in ord_cols:
            gk = [dff[dff[col]==v]["price"].values for v in sorted(dff[col].unique()) if len(dff[dff[col]==v])>0]
            if len(gk)<2: continue
            stat_k, p_k = stats.kruskal(*gk)
            test_rows.append({
                "Variable":         col,
                "Prix sans (M)":    "—",
                "Prix avec (M)":    "—",
                "Différence (%)":   "—",
                "Test":             "Kruskal-Wallis",
                "Statistique U":    round(stat_k, 2),
                "p-value":          round(p_k, 5),
                "Significatif":     "Oui" if p_k<0.05 else "Non",
                "Taille effet r":   "—",
            })
        test_df = pd.DataFrame(test_rows)
        st.dataframe(test_df, use_container_width=True, hide_index=True)

    footer()


# PAGE : ANALYSE MULTIVARIÉE
elif page == "Analyse multivariée":
    st.markdown("## Analyse multivariée")
    st.caption("Interactions entre plusieurs variables simultanément")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Interactions binaires", "Heatmaps croisées",
        "Profils de segments", "Scatter Matrix"
    ])

    #  Tab 1 : contrôle strict des variables X ≠ Y 
    with tab1:
        st.markdown("### Interactions entre variables binaires")
        st.markdown("""<div class="warning-box">
            ℹ️ Le menu Variable Y exclut automatiquement la variable sélectionnée en Variable X
            pour éviter toute comparaison d'une variable avec elle-même.
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        # Variable X
        var_x = c1.selectbox(
            "Variable X (ligne)", binary_cols, key="mv_x",
            format_func=lambda x: EQUIP_LABELS.get(x, x)
        )
        # Variable Y : EXCLUT var_x dynamiquement
        var_y_opts = [v for v in binary_cols if v != var_x]
        var_y = c2.selectbox(
            "Variable Y (colonne)", var_y_opts, key="mv_y",
            format_func=lambda x: EQUIP_LABELS.get(x, x)
        )

        # Garde-fou final (ne devrait jamais se déclencher)
        if var_x == var_y:
            st.error("⛔ Variable X et Variable Y identiques — veuillez sélectionner deux variables distinctes.")
            st.stop()

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        pivot = dff.pivot_table(values="price_M", index=var_x, columns=var_y, aggfunc="mean")
        pivot.index   = ["Non" if i==0 else "Oui" for i in pivot.index]
        pivot.columns = ["Non" if c==0 else "Oui" for c in pivot.columns]
        pivot.plot(kind="bar", ax=axes[0], color=[BLUE,GREEN], edgecolor="white", rot=0)
        axes[0].set_title(f"Prix moyen : {EQUIP_LABELS[var_x]} × {EQUIP_LABELS[var_y]}")
        axes[0].set_ylabel("Prix moyen (millions)"); axes[0].legend(title=EQUIP_LABELS[var_y])
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt="%.2f", fontsize=8, padding=2)

        pivot2 = dff.pivot_table(values="price_M", index="furnishingstatus",
                                  columns="airconditioning", aggfunc="mean")
        pivot2.columns = ["Sans AC","Avec AC"]
        pivot2.plot(kind="bar", ax=axes[1], color=[BLUE,RED], edgecolor="white", rot=0)
        axes[1].set_title("Prix moyen : Ameublement × Climatisation")
        axes[1].set_ylabel("Prix moyen (millions)"); axes[1].legend(title="AC")
        for container in axes[1].containers:
            axes[1].bar_label(container, fmt="%.2f", fontsize=8, padding=2)

        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown(f"##### Tableau pivot — {EQUIP_LABELS[var_x]} × {EQUIP_LABELS[var_y]}")
        p_simple = dff.pivot_table(
            values="price_M", index=var_x, columns=var_y,
            aggfunc="mean", margins=True, margins_name="Total"
        )
        # Renommer les index et colonnes
        new_index = [("Non" if i==0 else "Oui") if i in [0,1] else str(i) for i in p_simple.index]
        new_cols  = [("Non" if c==0 else "Oui") if c in [0,1] else str(c) for c in p_simple.columns]
        p_simple.index   = new_index
        p_simple.columns = new_cols
        st.dataframe(p_simple.round(3), use_container_width=True)

        # Nombre de biens par combinaison
        st.markdown(f"##### Nb de propriétés par combinaison {EQUIP_LABELS[var_x]} × {EQUIP_LABELS[var_y]}")
        cnt_pivot = dff.pivot_table(values="price_M", index=var_x, columns=var_y, aggfunc="count")
        cnt_pivot.index   = ["Non" if i==0 else "Oui" for i in cnt_pivot.index]
        cnt_pivot.columns = ["Non" if c==0 else "Oui" for c in cnt_pivot.columns]
        st.dataframe(cnt_pivot, use_container_width=True)

    # Tab 2 : Heatmaps
    with tab2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        p1 = dff.pivot_table(values="price_M", index="bedrooms", columns="bathrooms", aggfunc="mean")
        sns.heatmap(p1, annot=True, fmt=".2f", cmap=SEQ, ax=axes[0], linewidths=.5, linecolor="white")
        axes[0].set_title("Prix moyen (M) : Chambres × Salles de bain")
        axes[0].set_xlabel("Salles de bain"); axes[0].set_ylabel("Chambres")

        p2 = dff.pivot_table(values="price_M", index="stories", columns="parking", aggfunc="mean")
        sns.heatmap(p2, annot=True, fmt=".2f", cmap=SEQ, ax=axes[1], linewidths=.5, linecolor="white")
        axes[1].set_title("Prix moyen (M) : Étages × Parking")
        axes[1].set_xlabel("Places de parking"); axes[1].set_ylabel("Étages")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown("##### Heatmaps — Nombre de propriétés")
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
        c1h = dff.pivot_table(values="price_M", index="bedrooms", columns="bathrooms", aggfunc="count")
        sns.heatmap(c1h, annot=True, fmt=".0f", cmap="YlOrRd", ax=axes2[0], linewidths=.5, linecolor="white")
        axes2[0].set_title("Nb propriétés : Chambres × Salles de bain")
        c2h = dff.pivot_table(values="price_M", index="stories", columns="parking", aggfunc="count")
        sns.heatmap(c2h, annot=True, fmt=".0f", cmap="YlOrRd", ax=axes2[1], linewidths=.5, linecolor="white")
        axes2[1].set_title("Nb propriétés : Étages × Parking")
        fig2.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close(fig2)

    # Tab 3 : Profils de segments
    with tab3:
        st.markdown("### Profil moyen par segment de prix")
        seg_profile = dff.groupby("price_segment", observed=False)[["area"]+binary_cols].mean()
        seg_norm = seg_profile.copy()
        seg_norm["area"] = seg_norm["area"] / seg_norm["area"].max()

        fig, ax = plt.subplots(figsize=(12, 4.5))
        x_sp = np.arange(len(seg_norm.columns))
        w = 0.28
        col_lbl = {
            "area":"Superficie (norm.)","mainroad":"Route princ.",
            "guestroom":"Ch. invités","basement":"Cave",
            "hotwaterheating":"Eau chaude","airconditioning":"Climatisation","prefarea":"Zone préf."
        }
        for i, (seg, color) in enumerate(zip(seg_norm.index, [BLUE,GREEN,NAVY])):
            ax.bar(x_sp + i*w, seg_norm.loc[seg], w, label=seg, color=color, edgecolor="white", alpha=.85)
        ax.set_xticks(x_sp+w)
        ax.set_xticklabels([col_lbl.get(c,c) for c in seg_norm.columns], rotation=20, ha="right")
        ax.set_ylabel("Score normalisé (0–1)"); ax.set_title("Profil des segments de prix", fontweight="bold")
        ax.legend(title="Segment"); ax.set_ylim(0, 1.15)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.markdown("##### Tableau de profil — Proportions (%)")
        pct_profile = (seg_profile[binary_cols]*100).round(1)
        pct_profile.insert(0, "Superficie moy. (ft²)", dff.groupby("price_segment", observed=False)["area"].mean().round(0))
        pct_profile.columns = [col_lbl.get(c,c) for c in pct_profile.columns]
        st.dataframe(pct_profile, use_container_width=True)

        show_insight("Les biens du segment 'Élevé' ont une probabilité d'avoir la climatisation et d'être en zone préférentielle significativement plus élevée que les autres segments.", "green")
        show_insight("La superficie normalisée croît linéairement avec le segment de prix — la taille reste un discriminant fort.", "green")

    # Tab 4 : Scatter Matrix
    with tab4:
        st.markdown("### Scatter Matrix — Variables quantitatives")
        sm_vars = ["price_M","area","bedrooms","bathrooms","stories"]
        n_sm = len(sm_vars)
        fig, axes_sm = plt.subplots(n_sm, n_sm, figsize=(13, 11))
        fig.suptitle("Scatter Matrix — Variables quantitatives", fontweight="bold", y=1.01)

        for i, rv in enumerate(sm_vars):
            for j, cv in enumerate(sm_vars):
                ax = axes_sm[i, j]
                if i == j:
                    ax.hist(dff[rv], bins=15, color=BLUE, edgecolor="white", lw=.4)
                    ax.set_facecolor("#FAFAFA")
                elif i > j:
                    ax.scatter(dff[cv], dff[rv], alpha=.2, s=6, color=BLUE, edgecolors="none")
                    ax.set_facecolor("#FAFAFA")
                else:
                    r_val = dff[[rv,cv]].corr().iloc[0,1]
                    ax.text(.5,.5, f"r={r_val:.2f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color=RED if abs(r_val)>.3 else NAVY,
                            transform=ax.transAxes)
                    ax.set_facecolor("white"); ax.set_xticks([]); ax.set_yticks([])
                if i==n_sm-1: ax.set_xlabel(rv, fontsize=7)
                if j==0:      ax.set_ylabel(rv, fontsize=7)
                ax.tick_params(labelsize=5)

        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

    footer()


# PAGE : INSIGHTS DÉCISIONNELS
elif page == "Insights décisionnels":
    st.markdown("## Insights décisionnels")
    st.caption("Analyses enrichies pour les décideurs — Vue exhaustive du marché immobilier")

    # Pré-calcul impact équipements
    impact_rows = []
    for col in binary_cols:
        g0_i = dff[dff[col]==0]["price"]; g1_i = dff[dff[col]==1]["price"]
        if len(g0_i)==0 or len(g1_i)==0: continue
        diff_i = (g1_i.mean()-g0_i.mean())/g0_i.mean()*100
        stat_i, p_i = stats.mannwhitneyu(g0_i, g1_i, alternative="two-sided")
        impact_rows.append({
            "Équipement":   EQUIP_LABELS[col],
            "Code":         col,
            "Prix sans (M)": round(g0_i.mean()/1e6, 3),
            "Prix avec (M)": round(g1_i.mean()/1e6, 3),
            "Gain absolu (M)": round((g1_i.mean()-g0_i.mean())/1e6, 3),
            "Écart (%)":    round(diff_i, 1),
            "p-value":      round(p_i, 4),
            "Significatif": "Oui" if p_i<0.05 else "Non",
            "Nb sans":      len(g0_i),
            "Nb avec":      len(g1_i),
        })
    impact_df = pd.DataFrame(impact_rows).sort_values("Écart (%)", ascending=False)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Impact équipements", "Simulateur de prix",
        "Analyse de marché", "ROI & Investissement", "Tableau exécutif"
    ])

    # Tab 1 : Impact
    with tab1:
        st.markdown("### Impact des équipements sur le prix")

        fig, ax = plt.subplots(figsize=(10, 4))
        colors_b = [GREEN if v>=0 else RED for v in impact_df["Écart (%)"]]
        bars_i = ax.barh(impact_df["Équipement"], impact_df["Écart (%)"], color=colors_b, edgecolor="white")
        ax.axvline(0, color="black", lw=.8)
        for bar, v in zip(bars_i, impact_df["Écart (%)"]):
            ax.text(v+(0.3 if v>=0 else -0.3), bar.get_y()+bar.get_height()/2,
                    f"{v:+.1f}%", va="center", ha="left" if v>=0 else "right",
                    fontsize=9.5, fontweight="bold", color=NAVY)
        ax.set_title("Différence de prix — avec vs sans l'équipement", fontweight="bold")
        ax.set_xlabel("Variation de prix (%)")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        st.dataframe(impact_df.drop(columns=["Code"]), use_container_width=True, hide_index=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### ✅ Leviers de valorisation")
            show_insight("La présence d'une route principale augmente le prix de +46.9% en moyenne — facteur d'accessibilité le plus déterminant.", "green")
            show_insight("La climatisation génère un écart de +43.4% — équipement hautement valorisé par le marché.", "green")
            show_insight("Être en zone résidentielle préférentielle apporte +32.9% par rapport aux zones ordinaires.", "green")
            show_insight("Une chambre d'invités supplémentaire valorise le bien de +27.5% en prix moyen.", "green")
        with col2:
            st.markdown("### ⚠️ Points de vigilance")
            show_insight("Le CV du prix est de 39.2% — marché très hétérogène, les prix sont très dispersés.", "red")
            show_insight("Le parking génère un faible différentiel de prix ; il ne constitue pas un argument de vente décisif seul.", "red")
            show_insight("Les propriétés à 4–6 chambres sont rares (< 10% du marché) ; leurs prix peuvent être biaisés par un petit effectif.", "red")
            show_insight("Skewness du prix = 1.73 — quelques biens de très haute valeur tirent la moyenne vers le haut.", "red")

    # ── Tab 2 : Simulateur ────────────────────────────────────────────────────
    with tab2:
        st.markdown("### Simulateur de prix — Estimation rapide")
        st.markdown("""<div class="warning-box">
            ℹ️ Estimation basée sur la régression linéaire superficie→prix et les différentiels
            observés par équipement. Il ne s'agit pas d'un modèle prédictif formel.
        </div>""", unsafe_allow_html=True)

        sc1, sc2 = st.columns(2)
        with sc1:
            s_area     = st.slider("Superficie (sq ft)", int(dff["area"].min()), int(dff["area"].max()),
                                   int(dff["area"].median()), step=50)
            s_beds     = st.select_slider("Chambres",       options=sorted(dff["bedrooms"].unique()),   value=3)
            s_baths    = st.select_slider("Salles de bain", options=sorted(dff["bathrooms"].unique()),  value=1)
            s_stories  = st.select_slider("Étages",         options=sorted(dff["stories"].unique()),    value=1)
            s_parking  = st.select_slider("Parking",        options=sorted(dff["parking"].unique()),    value=0)
        with sc2:
            s_ac       = st.checkbox("Climatisation",       value=True)
            s_prefarea = st.checkbox("Zone préférentielle", value=False)
            s_mainroad = st.checkbox("Route principale",    value=True)
            s_guestroom= st.checkbox("Chambre d'invités",  value=False)
            s_basement = st.checkbox("Cave / sous-sol",    value=False)
            s_hotwater = st.checkbox("Eau chaude",         value=False)

        m_s, b_s, _, _, _ = stats.linregress(dff["area"], dff["price"])
        base = m_s * s_area + b_s
        mult = 1.0
        feature_map = {
            "airconditioning": s_ac, "prefarea": s_prefarea,
            "mainroad": s_mainroad, "guestroom": s_guestroom,
            "basement": s_basement, "hotwaterheating": s_hotwater,
        }
        for code, active in feature_map.items():
            row = impact_df[impact_df["Code"]==code]
            if active and len(row)>0:
                mult *= (1 + row["Écart (%)"].values[0]/100)

        estimated = base * mult
        pct_rank = (dff["price"] < estimated).mean() * 100

        mask_comp = (
            (dff["airconditioning"]==int(s_ac)) &
            (dff["prefarea"]==int(s_prefarea)) &
            (dff["bedrooms"]==s_beds)
        )
        comp = dff[mask_comp]

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        sr1,sr2,sr3,sr4 = st.columns(4)
        sr1.markdown(metric_card("Prix estimé", f"{estimated/1e6:.3f} M", "estimation modèle"), unsafe_allow_html=True)
        if len(comp)>0:
            sr2.markdown(metric_card("Médiane comparable", f"{comp['price'].median()/1e6:.3f} M", f"{len(comp)} biens similaires"), unsafe_allow_html=True)
            sr3.markdown(metric_card("Fourchette comparable", f"{comp['price'].min()/1e6:.2f}–{comp['price'].max()/1e6:.2f} M", "min–max"), unsafe_allow_html=True)
        else:
            sr2.markdown(metric_card("Médiane comparable", "—", "Aucun comparable"), unsafe_allow_html=True)
            sr3.markdown(metric_card("Fourchette", "—", "—"), unsafe_allow_html=True)
        sr4.markdown(metric_card("Percentile marché", f"P{pct_rank:.0f}", "rang dans le marché"), unsafe_allow_html=True)

        show_insight(f"Le bien simulé se situe au {pct_rank:.1f}e percentile du marché — "
                     f"{'au-dessus de la médiane' if pct_rank>50 else 'en-dessous de la médiane'}.", "orange")

    #  Tab 3 : Analyse de marché 
    with tab3:
        st.markdown("### Analyse de marché — Distribution et segmentation")

        ma1, ma2 = st.columns(2)
        with ma1:
            furn_stats = dff.groupby("furnishingstatus")["price_M"].agg(
                Nb="count", Moyenne="mean", Médiane="median", Min="min", Max="max", Std="std"
            ).round(3)
            st.markdown("#### Par ameublement")
            st.dataframe(furn_stats, use_container_width=True)
        with ma2:
            seg_stats = dff.groupby("price_segment", observed=False)["price_M"].agg(
                Nb="count", Moyenne="mean", Médiane="median", Min="min", Max="max"
            ).round(3)
            st.markdown("#### Par segment de prix")
            st.dataframe(seg_stats, use_container_width=True)

        # Violin + quintile
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        groups_f = [dff[dff["furnishingstatus"]==v]["price_M"].values
                    for v in ["furnished","semi-furnished","unfurnished"]]
        vp = axes[0].violinplot(groups_f, positions=[1,2,3], showmedians=True, showmeans=True)
        for pc, color in zip(vp["bodies"], [BLUE,GREEN,NAVY]):
            pc.set_facecolor(color); pc.set_alpha(.55)
        axes[0].set_xticks([1,2,3]); axes[0].set_xticklabels(["Meublé","Semi-meublé","Non meublé"])
        axes[0].set_ylabel("Prix (millions)"); axes[0].set_title("Distribution prix par ameublement (Violin)")

        dff["area_q"] = pd.qcut(dff["area"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
        aq = dff.groupby("area_q", observed=False)["price_M"].agg(["mean","median","count"]).reset_index()
        axes[1].bar(aq["area_q"], aq["mean"], color=BLUE, alpha=.8, edgecolor="white", label="Moyenne")
        axes[1].plot(range(len(aq)), aq["median"], "D-", color=RED, markersize=7, label="Médiane")
        axes[1].set_xlabel("Quintile de superficie"); axes[1].set_ylabel("Prix (millions)")
        axes[1].set_title("Prix moyen par quintile de superficie"); axes[1].legend(fontsize=8)
        for i, (bar, row) in enumerate(zip(axes[1].patches, aq.itertuples())):
            axes[1].text(i, bar.get_height()+.02, f"n={row.count}", ha="center", va="bottom", fontsize=7)
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Courbe de Lorenz
        st.markdown("#### Concentration de marché — Courbe de Lorenz")
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.5))
        sorted_p = np.sort(dff["price"].values)
        cum_pop = np.arange(1,len(sorted_p)+1)/len(sorted_p)
        cum_val = np.cumsum(sorted_p)/sorted_p.sum()
        gini = 1 - 2*np.trapezoid(cum_val, cum_pop)
        axes2[0].plot(cum_pop, cum_val, color=BLUE, lw=2, label=f"Lorenz (Gini={gini:.3f})")
        axes2[0].plot([0,1],[0,1], "k--", lw=1, label="Égalité parfaite")
        axes2[0].fill_between(cum_pop, cum_pop, cum_val, alpha=.12, color=BLUE)
        axes2[0].set_xlabel("Part cumulée des propriétés"); axes2[0].set_ylabel("Part cumulée de la valeur")
        axes2[0].set_title("Courbe de Lorenz — Concentration des prix"); axes2[0].legend(fontsize=8)

        percs_m = [10,25,50,75,90,95,99]
        vals_m  = [dff["price"].quantile(p/100)/1e6 for p in percs_m]
        lbls_m  = [f"P{p}" for p in percs_m]
        colors_m = [BLUE,BLUE,GREEN,GREEN,ORANGE,ORANGE,RED]
        axes2[1].bar(lbls_m, vals_m, color=colors_m, edgecolor="white")
        axes2[1].set_ylabel("Prix (millions)"); axes2[1].set_title("Percentiles de prix du marché")
        for bar, v in zip(axes2[1].patches, vals_m):
            axes2[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02,
                          f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        fig2.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close(fig2)

        show_insight(f"Indice de Gini = {gini:.3f} — concentration modérée. Les biens les plus chers (top 10%) détiennent une part disproportionnée de la valeur totale du marché.", "orange")
        perc_df_m = pd.DataFrame({"Percentile": lbls_m, "Prix (M)": [round(v,3) for v in vals_m]})

    # Tab 4 : ROI & Investissement
    with tab4:
        st.markdown("### ROI & Analyse d'investissement")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        impact_abs = impact_df.sort_values("Gain absolu (M)", ascending=False)
        colors_ga = [GREEN if v>=0 else RED for v in impact_abs["Gain absolu (M)"]]
        axes[0].barh(impact_abs["Équipement"], impact_abs["Gain absolu (M)"], color=colors_ga, edgecolor="white")
        axes[0].axvline(0, color="black", lw=.8)
        axes[0].set_title("Gain absolu sur le prix par équipement (M)", fontweight="bold")
        axes[0].set_xlabel("Gain (millions)")
        for bar, v in zip(axes[0].patches, impact_abs["Gain absolu (M)"]):
            axes[0].text(v+.005, bar.get_y()+bar.get_height()/2,
                         f"+{v:.3f}M" if v>=0 else f"{v:.3f}M", va="center", ha="left", fontsize=8.5)

        ac_seg = dff.groupby(["price_segment","airconditioning"], observed=False).size().unstack(fill_value=0)
        ac_seg.columns = ["Sans AC","Avec AC"]
        ac_seg.plot(kind="bar", ax=axes[1], color=[BLUE,GREEN], edgecolor="white", rot=0)
        axes[1].set_title("Climatisation par segment de prix")
        axes[1].set_xlabel("Segment de prix"); axes[1].set_ylabel("Nb propriétés")
        axes[1].legend(title="AC")
        fig.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close(fig)

        # Score d'investissement
        st.markdown("#### Score d'investissement — Cumul d'équipements valorisants")
        dff["premium_score"] = dff[["airconditioning","prefarea","mainroad","guestroom"]].sum(axis=1)
        score_stats = dff.groupby("premium_score")["price_M"].agg(
            Nb="count", Moyenne="mean", Médiane="median", Std="std"
        ).round(3)
        score_stats.index.name = "Score (0–4 équipements)"

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        colors_sc = [BLUE,GREEN,ORANGE,RED,PURPLE]
        bars_sc = ax3.bar(score_stats.index, score_stats["Moyenne"],
                          color=colors_sc[:len(score_stats)], edgecolor="white", alpha=.85)
        ax3.plot(range(len(score_stats)), score_stats["Médiane"], "D-",
                 color=NAVY, markersize=8, label="Médiane")
        ax3.set_xlabel("Score investissement (nb équipements valorisants)")
        ax3.set_ylabel("Prix moyen (millions)")
        ax3.set_title("Prix selon le score d'investissement d'équipements", fontweight="bold")
        ax3.legend(fontsize=9)
        for bar, row in zip(bars_sc, score_stats.itertuples()):
            ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.02,
                     f"n={row.Nb}", ha="center", va="bottom", fontsize=8)
        fig3.tight_layout(); st.pyplot(fig3, use_container_width=True); plt.close(fig3)

        st.dataframe(score_stats, use_container_width=True)

        show_insight("Chaque équipement premium ajouté (AC, zone préf., route princ., ch. invités) accroît le prix de manière quasi-linéaire.", "green")
        show_insight("Un bien avec les 4 équipements premium peut valoir 2× à 3× un bien sans aucun — potentiel de revalorisation par rénovation ciblée.", "green")
        show_insight("Stratégie d'arbitrage : acquérir des biens score 0–1 en zone à potentiel route principale, installer AC et réorienter en zone préférentielle.", "purple")

    # Tab 5 : Tableau de bord exécutif
    with tab5:
        st.markdown("### Tableau de bord exécutif")

        e1,e2,e3,e4 = st.columns(4)
        e1.markdown(metric_card("Valeur totale marché", f"{dff['price'].sum()/1e9:.2f} Mds", "milliards"), unsafe_allow_html=True)
        e2.markdown(metric_card("Valeur moy. bien",     f"{dff['price'].mean()/1e6:.3f} M", "par propriété"), unsafe_allow_html=True)
        e3.markdown(metric_card("Biens premium (AC+Zone)", f"{(dff[['airconditioning','prefarea']].sum(axis=1)==2).sum()}", "AC + Zone préférentielle"), unsafe_allow_html=True)
        e4.markdown(metric_card("Levier max.",          f"+{impact_df['Écart (%)'].max():.1f}%", impact_df.loc[impact_df['Écart (%)'].idxmax(),'Équipement']), unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        fig_e, axes_e = plt.subplots(2, 2, figsize=(13, 9))
        fig_e.suptitle("Vue exécutive du marché immobilier", fontweight="bold", fontsize=13)

        # 1. Valeur totale par segment
        sv = dff.groupby("price_segment", observed=False)["price"].sum().reindex(["Bas","Moyen","Élevé"])/1e9
        axes_e[0,0].bar(sv.index, sv.values, color=[BLUE,GREEN,NAVY], edgecolor="white")
        axes_e[0,0].set_title("Valeur totale par segment (Milliards)")
        axes_e[0,0].set_ylabel("Milliards")
        for bar, v in zip(axes_e[0,0].patches, sv.values):
            axes_e[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+.01,
                             f"{v:.2f} Md", ha="center", va="bottom", fontsize=9, fontweight="bold")

        # 2. Impact équipements (horizontal)
        axes_e[0,1].barh(impact_df["Équipement"], impact_df["Écart (%)"], color=GREEN, edgecolor="white", alpha=.85)
        axes_e[0,1].axvline(0, color="black", lw=.8)
        axes_e[0,1].set_title("Équipements — Impact sur le prix (%)")
        axes_e[0,1].set_xlabel("Variation (%)")
        for bar, v in zip(axes_e[0,1].patches, impact_df["Écart (%)"]):
            axes_e[0,1].text(v+.5, bar.get_y()+bar.get_height()/2,
                             f"{v:+.1f}%", va="center", ha="left", fontsize=8)

        # 3. Prix par nombre de chambres
        bed_s = dff.groupby("bedrooms")["price_M"].agg(["mean","median","count"]).reset_index()
        axes_e[1,0].plot(bed_s["bedrooms"], bed_s["mean"],   "o-", color=BLUE, lw=2, ms=8, label="Moyenne")
        axes_e[1,0].plot(bed_s["bedrooms"], bed_s["median"], "s--",color=RED,  lw=2, ms=7, label="Médiane")
        axes_e[1,0].fill_between(bed_s["bedrooms"], bed_s["mean"], bed_s["median"], alpha=.1, color=BLUE)
        axes_e[1,0].set_xlabel("Nb de chambres"); axes_e[1,0].set_ylabel("Prix (millions)")
        axes_e[1,0].set_title("Prix moyen/médian par nb de chambres"); axes_e[1,0].legend(fontsize=8)

        # 4. Combinaisons AC × Zone préférentielle
        combos = [
            (dff[(dff["airconditioning"]==0)&(dff["prefarea"]==0)], "Basique",          BLUE),
            (dff[(dff["airconditioning"]==1)&(dff["prefarea"]==0)], "AC seul",           GREEN),
            (dff[(dff["airconditioning"]==0)&(dff["prefarea"]==1)], "Zone préf. seule",  ORANGE),
            (dff[(dff["airconditioning"]==1)&(dff["prefarea"]==1)], "AC + Zone préf.",   RED),
        ]
        lbl_c4 = [c[1] for c in combos]
        val_c4 = [c[0]["price_M"].mean() if len(c[0])>0 else 0 for c in combos]
        cnt_c4 = [len(c[0]) for c in combos]
        clr_c4 = [c[2] for c in combos]
        axes_e[1,1].bar(range(4), val_c4, color=clr_c4, edgecolor="white", alpha=.85)
        axes_e[1,1].set_xticks(range(4)); axes_e[1,1].set_xticklabels(lbl_c4, rotation=15, ha="right", fontsize=8)
        axes_e[1,1].set_ylabel("Prix moyen (millions)")
        axes_e[1,1].set_title("Prix moyen — Combinaisons AC × Zone préf.")
        for i, (v, n) in enumerate(zip(val_c4, cnt_c4)):
            axes_e[1,1].text(i, v+.02, f"{v:.2f}\nn={n}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        fig_e.tight_layout(); st.pyplot(fig_e, use_container_width=True); plt.close(fig_e)

        # Recommandations stratégiques
        st.markdown("#### Recommandations stratégiques")
        rc1, rc2 = st.columns(2)
        with rc1:
            show_insight("PRIORITÉ 1 — Route principale (+46.9%) : critère non-négociable pour la valorisation maximale d'un bien.", "green")
            show_insight("PRIORITÉ 2 — Climatisation (+43.4%) : ROI quasi-garanti dans un marché où moins de 40% des biens en sont équipés.", "green")
            show_insight("PRIORITÉ 3 — Zone préférentielle (+32.9%) : facteur structurel inamovible, à privilégier à l'achat.", "green")
        with rc2:
            show_insight("RISQUE — Marché hétérogène (CV=39.2%) : la dispersion des prix exige une analyse de biens comparables avant toute transaction.", "red")
            show_insight("RISQUE — Asymétrie de distribution (Skew=1.73) : la moyenne surestime la valeur typique. Utiliser la médiane comme référence.", "red")
            show_insight("OPPORTUNITÉ — Acquérir des biens score 0–1 et les rénover (AC + aménagements) pour capturer la prime de valeur.", "purple")

    footer()

# PAGE : EXPORTS GLOBAUX
elif page == "Exports globaux":
    st.markdown("## Exports globaux — Téléchargement des analyses")
    st.caption("Téléchargez tous les tableaux analytiques importants au format CSV")

    st.markdown("""<div class="warning-box">
        ℹ️ Les données reflètent les filtres actifs dans la barre latérale.
        Chaque tableau est également sauvegardé dans le répertoire <code>tables/</code>.
    </div>""", unsafe_allow_html=True)

    # Génération de tous les tableaux
    # 1. Statistiques descriptives
    desc_e = dff[num_cols].describe().T.round(5)
    desc_e.columns = ["Nb","Moyenne","Écart-type","Min","Q1 (25%)","Médiane","Q3 (75%)","Max"]
    desc_e["CV (%)"]   = (desc_e["Écart-type"]/desc_e["Moyenne"]*100).round(2)
    desc_e["Skewness"] = [round(dff[c].skew(),4)     for c in num_cols]
    desc_e["Kurtosis"] = [round(dff[c].kurtosis(),4) for c in num_cols]

    # 2. Qualité des données
    qual_e = pd.DataFrame({
        "Type":               df.dtypes,
        "Non-null":           df.notnull().sum(),
        "Valeurs manquantes": df.isnull().sum(),
        "Missing (%)":        (df.isnull().mean()*100).round(2),
        "Valeurs uniques":    df.nunique(),
        "Min":                df.min(),
        "Max":                df.max(),
    })

    # 3. Corrélations (quantitatives uniquement)
    corr_e = dff[num_cols].corr().round(5)

    # 4. Impact équipements
    imp_rows_e = []
    for col in binary_cols:
        g0_e = dff[dff[col]==0]["price"]; g1_e = dff[dff[col]==1]["price"]
        if len(g0_e)==0 or len(g1_e)==0: continue
        diff_e = (g1_e.mean()-g0_e.mean())/g0_e.mean()*100
        stat_e, p_e = stats.mannwhitneyu(g0_e, g1_e, alternative="two-sided")
        imp_rows_e.append({
            "Équipement":    EQUIP_LABELS[col],
            "Prix sans (M)": round(g0_e.mean()/1e6, 3),
            "Prix avec (M)": round(g1_e.mean()/1e6, 3),
            "Gain abs. (M)": round((g1_e.mean()-g0_e.mean())/1e6, 3),
            "Écart (%)":     round(diff_e, 1),
            "p-value":       round(p_e, 5),
            "Significatif":  "Oui" if p_e<0.05 else "Non",
        })
    imp_e = pd.DataFrame(imp_rows_e).sort_values("Écart (%)", ascending=False)

    # 5. Tests statistiques
    tst_rows_e = []
    for col in binary_cols:
        g0_t2 = dff[dff[col]==0]["price"]; g1_t2 = dff[dff[col]==1]["price"]
        if len(g0_t2)==0 or len(g1_t2)==0: continue
        stat_t2, p_t2 = stats.mannwhitneyu(g0_t2, g1_t2, alternative="two-sided")
        tst_rows_e.append({
            "Variable":      EQUIP_LABELS[col],
            "Test":          "Mann-Whitney",
            "Prix sans (M)": round(g0_t2.mean()/1e6, 3),
            "Prix avec (M)": round(g1_t2.mean()/1e6, 3),
            "Différence (%)":round((g1_t2.mean()-g0_t2.mean())/g0_t2.mean()*100, 1),
            "p-value":       round(p_t2, 5),
            "Significatif":  "Oui" if p_t2<0.05 else "Non",
        })
    tst_e = pd.DataFrame(tst_rows_e)

    # 6. Profil segments
    seg_pe = dff.groupby("price_segment", observed=False)[["area"]+binary_cols].mean().round(3)
    seg_pe.index.name = "Segment"

    # 7. Score investissement
    dff["premium_score"] = dff[["airconditioning","prefarea","mainroad","guestroom"]].sum(axis=1)
    prem_e = dff.groupby("premium_score")["price_M"].agg(
        Nb="count", Moyenne="mean", Médiane="median", Std="std"
    ).round(3)

    # 8. Distribution par ameublement
    furn_e = dff.groupby("furnishingstatus")["price_M"].agg(
        Nb="count", Moyenne="mean", Médiane="median", Min="min", Max="max", Std="std"
    ).round(3)

    # 9. Percentiles de prix
    pct_vals_e = [1,5,10,25,50,75,90,95,99]
    perc_e = pd.DataFrame({
        "Percentile (%)": pct_vals_e,
        "Prix (M)":       [round(dff["price"].quantile(p/100)/1e6, 3) for p in pct_vals_e],
    })

    # 10. Tests de normalité
    norm_rows_e = []
    for col in ["price","area"]:
        d_c = dff[col]
        sw_s, sw_p = stats.shapiro(d_c.sample(min(len(d_c),5000), random_state=42))
        jb_s, jb_p = stats.jarque_bera(d_c)
        norm_rows_e.append({
            "Variable": col, "Skewness": round(d_c.skew(),4),
            "Shapiro-Wilk p": round(sw_p,6), "Jarque-Bera p": round(jb_p,6),
            "Normalité rejetée": "Oui" if sw_p<0.05 or jb_p<0.05 else "Non",
        })
    norm_e = pd.DataFrame(norm_rows_e)

    # ── Dictionnaire des tableaux 
    tables_dict = {
        "Statistiques descriptives complètes":          ("stats_descriptives.csv",    desc_e),
        "Qualité des données":                          ("E02_qualite_donnees.csv",        qual_e),
        "Matrice de corrélation (quantitatives seules)":("E03_correlation_quantitatives.csv", corr_e),
        "Impact des équipements sur le prix":           ("E04_impact_equipements.csv",     imp_e),
        "Tests statistiques (Mann-Whitney)":            ("E05_tests_statistiques.csv",     tst_e),
        "Profil des segments de prix":                  ("E06_profil_segments.csv",        seg_pe),
        "Score d'investissement — Analyse ROI":                  ("E07_roi_premium_score.csv",      prem_e),
        "Distribution par ameublement":                 ("E08_stats_ameublement.csv",      furn_e),
        "Percentiles de prix du marché":                ("E09_percentiles_prix.csv",       perc_e),
        "Tests de normalité":                           ("E10_tests_normalite.csv",        norm_e),
    }

    for i, (name, (filename, df_t)) in enumerate(tables_dict.items()):
        with st.expander(name, expanded=(i == 0)):
            st.dataframe(df_t, use_container_width=True)
            st.download_button(
                label=f"⬇️ Télécharger — {filename}",
                data=to_csv_bytes(df_t),
                file_name=filename,
                mime="text/csv",
                key=f"dl_{i}",
            )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### Rapport de sauvegarde et Téléchargement groupé")

    combined = ""
    for name, (filename, df_t) in tables_dict.items():
        combined += f"\n\n{'='*60}\n{name}\n{'='*60}\n"
        combined += df_t.to_csv()

    st.download_button(
        label="⬇️ Télécharger TOUS les tableaux (rapport complet .csv)",
        data=combined.encode("utf-8"),
        file_name="housing_analytics_rapport_complet.csv",
        mime="text/csv",
        key="dl_all",
    )

    st.success(f"{len(tables_dict)} tableaux disponibles — sauvegardés dans le répertoire `/outputs/tables/`.")

    footer()
