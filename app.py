import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import re
import os

# ==========================================
# 0. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Country Health Figures", layout="wide")
sns.set_theme(style="whitegrid", font_scale=1.1)

# --- Define Categories ---
HVI_NCD_COLS = [
    "Cardiovascular diseases", "Chronic respiratory diseases", "Diabetes and kidney diseases",
    "Neoplasms", "Neurological disorders", "Digestive diseases", "Musculoskeletal disorders"
]

HVI_ID_COLS = [
    "Respiratory infections and tuberculosis", "Enteric infections", "Neglected tropical diseases and malaria",
    "Other infectious diseases", "Maternal and neonatal disorders", "Nutritional deficiencies"
]

DIET_HELPFUL_COLS = [
    "Fruits", "Non-starchy vegetables", "Beans and legumes", "Nuts and seeds",
    "Whole grains", "Total seafoods", "Yoghurt (including fermented milk)", "Coffee", "Tea"
]

DIET_HARMFUL_COLS = [
    "Sugar-sweetened beverages", "Refined grains", "Total processed meats", "Fruit juices"
]

DIET_NEUTRAL_COLS = [
    "Potatoes", "Other starchy vegetables", "Eggs", "Cheese", "Total Milk"
]

# --- Color Palettes ---
GBD_PAL = {
    "GBD1": "#e7298a", "GBD2": "#66a61e", "GBD3": "#1b9e77", "GBD4": "#a6761d", "GBD5": "#7570b3"
}

PHDI_PAL = {
    "PHDI1": "#66c2a5", "PHDI2": "#fc8d62", "PHDI3": "#8da0cb", "PHDI_total": "#000000"
}

CLIM_PAL = {
    "MeanT": "#1f78b4", "Heat_index": "#33a02c", "PM": "#6a3d9a", "RH": "#1b9e77", "Precipitation": "#a6cee3"
}

ECON_PAL = {
    "Gdp": "#e31a1c", "Trade": "#ff7f00", "Urb": "#b15928"
}

HVI_PAL = {
    "NCD": "#4E79A7", "ID": "#59A14F", "Other": "#9C755F"
}

DIET_PAL = {
    "Helpful": "#1B9E77", "Harmful": "#D62728", "Neutral": "#9467BD"
}

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def pick_column(patterns, columns):
    """Finds the first column matching regex patterns."""
    for p in patterns:
        for col in columns:
            if re.search(p, col, re.IGNORECASE):
                return col
    return None

def calc_zscore(series):
    """Calculates Z-score (standardized value)."""
    if series.std() == 0:
        return series - series.mean()
    return (series - series.mean()) / series.std()

def get_trend(df, x_col, y_col):
    """Calculates linear slope and determines trend direction."""
    df = df.dropna(subset=[x_col, y_col])
    if len(df) < 2:
        return "flat"
    slope, _, _, _, _ = linregress(df[x_col], df[y_col])
    return "up" if slope >= 0 else "down"

def plot_with_trend(data, x, y, group, palette, title, ylabel, ylim=None):
    """Generic plotting function for trends (Solid=Up, Dashed=Down) + Arrow."""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    groups = data[group].unique()
    
    for g in groups:
        subset = data[data[group] == g].sort_values(by=x)
        if subset.empty: continue
        
        trend = get_trend(subset, x, y)
        linestyle = '-' if trend == 'up' else '--'
        color = palette.get(g, "#333333")
        
        ax.plot(subset[x], subset[y], label=g, color=color, linestyle=linestyle, linewidth=2)
        
        last_pt = subset.iloc[-1]
        marker = '▲' if trend == 'up' else '▼'
        ax.text(last_pt[x], last_pt[y], marker, color=color, fontsize=12, ha='left', va='center')

    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if ylim:
        ax.set_ylim(ylim)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
    
    return fig

# ==========================================
# 2. MAIN APP LOGIC
# ==========================================

# 1. Load Data directly from the container/repo
DATA_FILE = "data_merged_quarters.csv"

if not os.path.exists(DATA_FILE):
    st.error(f"File '{DATA_FILE}' not found. Please ensure it is uploaded to your GitHub repository.")
    st.stop()
else:
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

# 2. Sidebar Country Selection (Full Name)
st.sidebar.header("Settings")

if "Country" in df.columns:
    # Sort countries alphabetically for better UX
    countries = sorted(df['Country'].unique().astype(str))
    selected_country = st.sidebar.selectbox("Select Country", countries, index=0)
else:
    st.error("Column 'Country' not found in dataset. Cannot filter.")
    st.stop()

# 3. Filter Data
country_df = df[df['Country'] == selected_country].copy()
country_df['year'] = pd.to_numeric(country_df['year'], errors='coerce')

st.title(f"Country Figures: {selected_country}")

# ==========================================
# 3. GENERATE PLOTS & TABLES
# ==========================================

# ---------------------------------------------------------
# PLOT 1: GBD Clusters
# ---------------------------------------------------------
st.subheader("1. Trends in Dominant Disease Burden Regimes (GBD1–GBD5)")
gbd_cols = ["GBD1", "GBD2", "GBD3", "GBD4", "GBD5"]

if all(col in country_df.columns for col in gbd_cols):
    plot_df = country_df[["year"] + gbd_cols].copy()
    for col in gbd_cols:
        plot_df[col] = calc_zscore(plot_df[col])
    
    plot_df_long = plot_df.melt('year', var_name='disease', value_name='value')
    
    fig1 = plot_with_trend(plot_df_long, 'year', 'value', 'disease', GBD_PAL, 
                           f"Trends in Dominant Disease Burden Regimes ({selected_country})", "Z Score")
    st.pyplot(fig1)
    
    # Table 1
    table1_data = {
        "Label": ["GBD1", "GBD2", "GBD3", "GBD4", "GBD5"],
        "Meaning": [
            "Cardiometabolic burden", 
            "Chronic systemic and degenerative burden", 
            "Infectious and maternal–child burden", 
            "Mixed inflammatory and multisystem burden", 
            "Injury, substance use, and residual burden"
        ],
        "Dominant Disease Profile": [
            "Cardiovascular diseases, diabetes, kidney disease",
            "Chronic respiratory diseases, neoplasms, neurological disorders",
            "Respiratory infections, enteric infections, maternal and neonatal disorders",
            "Digestive diseases, musculoskeletal disorders, skin and subcutaneous diseases",
            "Substance use disorders, injuries, other non-communicable diseases"
        ]
    }
    st.table(pd.DataFrame(table1_data))
    
else:
    st.warning("GBD columns missing.")

# ---------------------------------------------------------
# PLOT 2: PHDI
# ---------------------------------------------------------
st.subheader("2. Population Health Diet Index (PHDI) Components and Total Score")
score_cols = [c for c in df.columns if c.startswith("Score_") and not c.endswith("_Q")]

phdi_map = {
    "PHDI1": ["Score_Whole_fruits", "Score_Nonstarchy_vegetables", "Score_Nuts_and_seeds", "Score_Legumes"],
    "PHDI2": ["Score_Whole_grains", "Score_Unsat_oils", "Score_Fish", "Score_Starchy_veg", "Score_Dairy", "Score_Red_meat", "Score_Eggs"],
    "PHDI3": ["Score_Sugar", "Score_Sat_fat"]
}

phdi_total_col = pick_column(["^PHDI[_\\. ]*total$", "PHDI_total", "PHDIscore", "^PHDI$"], df.columns)

phdi_plot_df = country_df.copy()

valid_phdi = True
for key, sub_cols in phdi_map.items():
    available_cols = [c for c in sub_cols if c in phdi_plot_df.columns]
    if available_cols:
        phdi_plot_df[key] = phdi_plot_df[available_cols].sum(axis=1)
    else:
        valid_phdi = False
        
if valid_phdi:
    if phdi_total_col and phdi_total_col in phdi_plot_df.columns:
        phdi_plot_df["PHDI_total"] = phdi_plot_df[phdi_total_col]
    else:
        # Fallback sum
        phdi_plot_df["PHDI_total"] = phdi_plot_df[[c for c in score_cols if c in phdi_plot_df.columns]].sum(axis=1)

    cols_to_plot = ["PHDI1", "PHDI2", "PHDI3", "PHDI_total"]
    phdi_long = phdi_plot_df[["year"] + cols_to_plot].melt("year", var_name="metric", value_name="value")
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=phdi_long, x="year", y="value", hue="metric", palette=PHDI_PAL, linewidth=2, ax=ax2)
    
    # Make Total thicker and black
    total_data = phdi_long[phdi_long['metric'] == "PHDI_total"]
    ax2.plot(total_data['year'], total_data['value'], color='black', linewidth=3, label='_nolegend_')
    
    ax2.set_title(f"PHDI Components and Total Score ({selected_country})", fontweight='bold')
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig2)
    
    # Table 2
    table2_data = {
        "Label": ["PHDI1", "PHDI2", "PHDI3"],
        "Meaning": [
            "Protective diet component",
            "Neutral/staple diet component",
            "Risk-enhancing diet component"
        ],
        "Dietary Interpretation": [
            "Fruits, vegetables, legumes, whole grains, nuts, seeds",
            "Milk, eggs, potatoes, starchy vegetables",
            "Refined grains, processed meats, sugar-sweetened beverages"
        ]
    }
    st.table(pd.DataFrame(table2_data))

else:
    st.warning("Missing PHDI component columns.")

# ---------------------------------------------------------
# PLOT 3: Climate
# ---------------------------------------------------------
st.subheader("3. Deviations in Environmental Stressors and Climate Variables")

cols = df.columns
clim_map = {
    "MeanT": pick_column(["^MeanT$", "mean[_\\. ]?temp", "temperature"], cols),
    "PM": pick_column(["PM2\\.5", "PM25", "^PM$"], cols),
    "RH": pick_column(["^RH$", "humidity"], cols),
    "Precipitation": pick_column(["^PR$", "precip", "rain"], cols),
    "Heat_index": pick_column(["^HI$", "heat[_\\. ]?index"], cols)
}

clim_map = {k: v for k, v in clim_map.items() if v is not None}

if clim_map:
    clim_df = country_df[["year"] + list(clim_map.values())].copy()
    inv_map = {v: k for k, v in clim_map.items()}
    clim_df.rename(columns=inv_map, inplace=True)
    
    plot_data = []
    for metric in clim_map.keys():
        z_score = calc_zscore(clim_df[metric])
        val_1990 = z_score[clim_df['year'] == 1990].values
        if len(val_1990) > 0:
            change = z_score - val_1990[0]
            temp_df = pd.DataFrame({'year': clim_df['year'], 'change': change, 'metric': metric})
            plot_data.append(temp_df)
    
    if plot_data:
        clim_long = pd.concat(plot_data)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=clim_long, x="year", y="change", hue="metric", palette=CLIM_PAL, linewidth=2, ax=ax3)
        ax3.set_title(f"Deviations in Environmental Stressors ({selected_country})", fontweight='bold')
        ax3.set_ylabel("Change from 1990 (Z-score)")
        ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig3)
        
        # Table 3
        table3_data = {
            "Variable": ["MeanT", "PM₂.₅ (PM)", "RH", "Precipitation (PR)", "Heat_index"],
            "Meaning": [
                "Mean annual temperature (°C)",
                "Population-weighted fine particulate matter",
                "Relative humidity (%)",
                "Annual accumulated rainfall (mm)",
                "Perceived temperature combining heat and humidity"
            ],
            "Health Relevance": [
                "Chronic heat load, ecosystem and labor stress",
                "Systemic inflammation, cardiopulmonary risk",
                "Modifies heat stress and infectious disease transmission",
                "Food security, drought/flood risk",
                "Human thermoregulatory stress and mortality risk"
            ]
        }
        st.table(pd.DataFrame(table3_data))
else:
    st.warning("Climate columns not found.")

# ---------------------------------------------------------
# PLOT 4: Economic
# ---------------------------------------------------------
st.subheader("4. Structural Buffering and Adaptive Capacity Indicators")
econ_map = {
    "Trade": pick_column(["^Trade$", "trade"], cols),
    "Gdp": pick_column(["^Gdp$", "gdp"], cols),
    "Urb": pick_column(["^Urb$", "urban"], cols)
}
econ_map = {k: v for k, v in econ_map.items() if v is not None}

if econ_map:
    econ_df = country_df[["year"] + list(econ_map.values())].copy()
    inv_map = {v: k for k, v in econ_map.items()}
    econ_df.rename(columns=inv_map, inplace=True)
    
    for col in econ_map.keys():
        econ_df[col] = calc_zscore(econ_df[col])
        
    econ_long = econ_df.melt("year", var_name="metric", value_name="z")
    
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=econ_long, x="year", y="z", hue="metric", palette=ECON_PAL, linewidth=2, ax=ax4)
    ax4.set_title(f"Structural Buffering and Adaptive Capacity ({selected_country})", fontweight='bold')
    ax4.set_ylabel("Z Score")
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig4)
    
    # Table 4
    table4_data = {
        "Variable": ["Trade", "GDP", "Urb"],
        "Meaning": [
            "Trade openness (% of GDP)",
            "GDP per capita (constant USD)",
            "Urbanization rate (%)"
        ],
        "Role in CARE-DDI": [
            "Food system connectivity and shock propagation",
            "Economic capacity for adaptation",
            "Infrastructure access, exposure concentration"
        ]
    }
    st.table(pd.DataFrame(table4_data))
    
else:
    st.warning("Economic columns not found.")

# ---------------------------------------------------------
# PLOT 5: HVI
# ---------------------------------------------------------
st.subheader("5. Trends in Health Vulnerability Domains (NCD vs. ID)")

hvi_df = country_df.copy()

has_ncd = [c for c in HVI_NCD_COLS if c in hvi_df.columns]
has_id = [c for c in HVI_ID_COLS if c in hvi_df.columns]

if has_ncd and has_id:
    all_disease_cols = has_ncd + has_id
    for c in all_disease_cols:
        hvi_df[c] = calc_zscore(hvi_df[c])
        
    hvi_df['NCD'] = hvi_df[has_ncd].mean(axis=1)
    hvi_df['ID'] = hvi_df[has_id].mean(axis=1)
    
    hvi_long = hvi_df[['year', 'NCD', 'ID']].melt('year', var_name='category', value_name='value')
    
    fig5 = plot_with_trend(hvi_long, 'year', 'value', 'category', HVI_PAL,
                           f"Trends in Health Vulnerability Domains ({selected_country})", "Z Score (Mean)")
    st.pyplot(fig5)
    
    # Table 5
    table5_data = {
        "Term": ["NCD", "ID"],
        "Meaning": [
            "Non-Communicable Diseases (e.g., cardiovascular, diabetes, cancer)",
            "Infectious Diseases, including maternal, neonatal, and nutritional disorders"
        ]
    }
    st.table(pd.DataFrame(table5_data))

# ---------------------------------------------------------
# PLOT 6: Diet Categories
# ---------------------------------------------------------
st.subheader("6. Dietary Adaptation Categories (Helpful, Harmful, and Neutral)")

diet_cols_all = DIET_HELPFUL_COLS + DIET_HARMFUL_COLS + DIET_NEUTRAL_COLS
valid_diet_cols = [c for c in diet_cols_all if c in country_df.columns]

if valid_diet_cols:
    diet_df = country_df[['year'] + valid_diet_cols].copy()
    
    for c in valid_diet_cols:
        diet_df[c] = calc_zscore(diet_df[c])
        
    if any(c in diet_df.columns for c in DIET_HELPFUL_COLS):
        diet_df['Helpful'] = diet_df[[c for c in DIET_HELPFUL_COLS if c in diet_df.columns]].mean(axis=1)
    if any(c in diet_df.columns for c in DIET_HARMFUL_COLS):
        diet_df['Harmful'] = diet_df[[c for c in DIET_HARMFUL_COLS if c in diet_df.columns]].mean(axis=1)
    if any(c in diet_df.columns for c in DIET_NEUTRAL_COLS):
        diet_df['Neutral'] = diet_df[[c for c in DIET_NEUTRAL_COLS if c in diet_df.columns]].mean(axis=1)
        
    diet_long = diet_df[['year', 'Helpful', 'Harmful', 'Neutral']].melt('year', var_name='category', value_name='value')
    
    fig6 = plot_with_trend(diet_long, 'year', 'value', 'category', DIET_PAL,
                           f"Dietary Adaptation Categories ({selected_country})", "Z Score (Mean)")
    st.pyplot(fig6)
    
    # Table 6
    table6_data = {
        "Category": ["Helpful", "Harmful", "Neutral"],
        "Meaning": [
            "Diets that reduce vulnerability and support resilience",
            "Diets that amplify disease risk, especially under heat and pollution",
            "Diets with context-dependent effects"
        ],
        "Examples": [
            "Fruits, vegetables, legumes, whole grains",
            "Processed meats, refined grains, sugary drinks",
            "Milk, eggs, potatoes, coffee, tea"
        ]
    }
    st.table(pd.DataFrame(table6_data))
