import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import re

# ==========================================
# 0. CONFIGURATION
# ==========================================
st.set_page_config(page_title="Country Health Figures", layout="wide")
sns.set_theme(style="whitegrid", font_scale=1.1)

# --- Define Categories (Mapped from R script) ---
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
    
    # Get unique groups
    groups = data[group].unique()
    
    for g in groups:
        subset = data[data[group] == g].sort_values(by=x)
        if subset.empty: continue
        
        # Determine trend
        trend = get_trend(subset, x, y)
        linestyle = '-' if trend == 'up' else '--'
        color = palette.get(g, "#333333")
        
        # Plot line
        ax.plot(subset[x], subset[y], label=g, color=color, linestyle=linestyle, linewidth=2)
        
        # Add arrow at the last point
        last_pt = subset.iloc[-1]
        marker = '▲' if trend == 'up' else '▼'
        ax.text(last_pt[x], last_pt[y], marker, color=color, fontsize=12, ha='left', va='center')

    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if ylim:
        ax.set_ylim(ylim)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicate labels if any
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    
    return fig

# ==========================================
# 2. MAIN APP LOGIC
# ==========================================

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload 'data_merged_quarters.csv'", type=['csv'])

if uploaded_file is not None:
    # Load Data
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("Data loaded successfully!")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Country Selection
    if "iso3" in df.columns:
        countries = df['iso3'].unique()
        selected_iso = st.sidebar.selectbox("Select Country (ISO3)", countries, index=0)
    else:
        st.error("Column 'iso3' not found in dataset.")
        st.stop()

    # Filter Data for Selected Country
    country_df = df[df['iso3'] == selected_iso].copy()
    country_df['year'] = pd.to_numeric(country_df['year'])

    st.title(f"Country Figures: {selected_iso}")

    # ---------------------------------------------------------
    # PLOT 1: Disease Trends (GBD1-GBD5)
    # ---------------------------------------------------------
    st.subheader("1. Disease Trends (GBD1–GBD5)")
    gbd_cols = ["GBD1", "GBD2", "GBD3", "GBD4", "GBD5"]
    
    if all(col in country_df.columns for col in gbd_cols):
        # Calculate Z-scores
        plot_df = country_df[["year"] + gbd_cols].copy()
        for col in gbd_cols:
            plot_df[col] = calc_zscore(plot_df[col])
        
        # Melt to long format
        plot_df_long = plot_df.melt('year', var_name='disease', value_name='value')
        
        fig1 = plot_with_trend(plot_df_long, 'year', 'value', 'disease', GBD_PAL, 
                               f"{selected_iso} GBD1 to GBD5 trends", "Z Score")
        st.pyplot(fig1)
    else:
        st.warning("GBD columns missing.")

    # ---------------------------------------------------------
    # PLOT 2: PHDI Lines
    # ---------------------------------------------------------
    st.subheader("2. PHDI Trends")
    # Identify Score columns
    score_cols = [c for c in df.columns if c.startswith("Score_") and not c.endswith("_Q")]
    
    # Calculate PHDI Components (if not already calculated or simple logic)
    # Mapping based on R script
    phdi_map = {
        "PHDI1": ["Score_Whole_fruits", "Score_Nonstarchy_vegetables", "Score_Nuts_and_seeds", "Score_Legumes"],
        "PHDI2": ["Score_Whole_grains", "Score_Unsat_oils", "Score_Fish", "Score_Starchy_veg", "Score_Dairy", "Score_Red_meat", "Score_Eggs"],
        "PHDI3": ["Score_Sugar", "Score_Sat_fat"]
    }
    
    # Determine Total PHDI column
    phdi_total_col = pick_column(["^PHDI[_\\. ]*total$", "PHDI_total", "PHDIscore", "^PHDI$"], df.columns)
    
    phdi_plot_df = country_df.copy()
    
    # Calculate sub-scores if columns exist
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
            # Fallback: sum all score cols
            phdi_plot_df["PHDI_total"] = phdi_plot_df[[c for c in score_cols if c in phdi_plot_df.columns]].sum(axis=1)

        cols_to_plot = ["PHDI1", "PHDI2", "PHDI3", "PHDI_total"]
        phdi_long = phdi_plot_df[["year"] + cols_to_plot].melt("year", var_name="metric", value_name="value")
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=phdi_long, x="year", y="value", hue="metric", palette=PHDI_PAL, linewidth=2, ax=ax2)
        
        # Make Total thicker and black (override)
        total_data = phdi_long[phdi_long['metric'] == "PHDI_total"]
        ax2.plot(total_data['year'], total_data['value'], color='black', linewidth=3, label='_nolegend_')
        
        ax2.set_title(f"PHDI Trends ({selected_iso})", fontweight='bold')
        st.pyplot(fig2)
    else:
        st.warning("Missing PHDI component columns.")

    # ---------------------------------------------------------
    # PLOT 3: Climate Change (Z-score change from 1990)
    # ---------------------------------------------------------
    st.subheader("3. Climate Change from 1990")
    
    cols = df.columns
    clim_map = {
        "MeanT": pick_column(["^MeanT$", "mean[_\\. ]?temp", "temperature"], cols),
        "PM": pick_column(["PM2\\.5", "PM25", "^PM$"], cols),
        "RH": pick_column(["^RH$", "humidity"], cols),
        "Precipitation": pick_column(["^PR$", "precip", "rain"], cols),
        "Heat_index": pick_column(["^HI$", "heat[_\\. ]?index"], cols)
    }
    
    # Filter out None
    clim_map = {k: v for k, v in clim_map.items() if v is not None}
    
    if clim_map:
        clim_df = country_df[["year"] + list(clim_map.values())].copy()
        
        # Rename columns to standardized keys
        inv_map = {v: k for k, v in clim_map.items()}
        clim_df.rename(columns=inv_map, inplace=True)
        
        # Calculate Z-scores then Change from 1990
        plot_data = []
        for metric in clim_map.keys():
            # Z-score
            z_score = calc_zscore(clim_df[metric])
            # Value at 1990
            val_1990 = z_score[clim_df['year'] == 1990].values
            if len(val_1990) > 0:
                change = z_score - val_1990[0]
                temp_df = pd.DataFrame({'year': clim_df['year'], 'change': change, 'metric': metric})
                plot_data.append(temp_df)
        
        if plot_data:
            clim_long = pd.concat(plot_data)
            fig3, ax3 = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=clim_long, x="year", y="change", hue="metric", palette=CLIM_PAL, linewidth=2, ax=ax3)
            ax3.set_title(f"Climate change from 1990 ({selected_iso})", fontweight='bold')
            ax3.set_ylabel("Change from 1990 (Z-score)")
            st.pyplot(fig3)
    else:
        st.warning("Climate columns not found.")

    # ---------------------------------------------------------
    # PLOT 4: Economic Indicators
    # ---------------------------------------------------------
    st.subheader("4. Economic Indicators")
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
        
        # Z-scores
        for col in econ_map.keys():
            econ_df[col] = calc_zscore(econ_df[col])
            
        econ_long = econ_df.melt("year", var_name="metric", value_name="z")
        
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=econ_long, x="year", y="z", hue="metric", palette=ECON_PAL, linewidth=2, ax=ax4)
        ax4.set_title(f"Economic Indicators ({selected_iso})", fontweight='bold')
        ax4.set_ylabel("Z Score")
        st.pyplot(fig4)
    else:
        st.warning("Economic columns not found.")

    # ---------------------------------------------------------
    # PLOT 5: HVI Disease Categories
    # ---------------------------------------------------------
    st.subheader("5. HVI Disease Categories")
    # Identify 'Other' columns (columns in data but not in NCD or ID lists)
    # Assuming disease columns start around index 25 in original data, 
    # but let's just check intersection with lists
    
    hvi_df = country_df.copy()
    
    # Check if columns exist
    has_ncd = [c for c in HVI_NCD_COLS if c in hvi_df.columns]
    has_id = [c for c in HVI_ID_COLS if c in hvi_df.columns]
    
    # Heuristic to find 'Other' columns (all columns that look like disease cols but aren't NCD/ID)
    # Since we can't perfectly know 'Other' without the specific dataset structure index,
    # we will rely on provided lists. If you have specific 'Other' cols, define them.
    # Here, we will calculate NCD and ID. 'Other' will be calculated if we can define the rest.
    # For now, let's plot NCD and ID, and try to deduce Other if possible or skip.
    
    if has_ncd and has_id:
        # Normalize all first
        all_disease_cols = has_ncd + has_id
        # Note: In R script, it took cols 26:43. We will just use the named lists.
        
        # Z-score normalization
        for c in all_disease_cols:
            hvi_df[c] = calc_zscore(hvi_df[c])
            
        hvi_df['NCD'] = hvi_df[has_ncd].mean(axis=1)
        hvi_df['ID'] = hvi_df[has_id].mean(axis=1)
        # Placeholder for Other - skipping for safety unless explicitly defined
        
        hvi_long = hvi_df[['year', 'NCD', 'ID']].melt('year', var_name='category', value_name='value')
        
        fig5 = plot_with_trend(hvi_long, 'year', 'value', 'category', HVI_PAL,
                               f"HVI Disease Categories ({selected_iso})", "Z Score (Mean)")
        st.pyplot(fig5)

    # ---------------------------------------------------------
    # PLOT 6: Diet Categories
    # ---------------------------------------------------------
    st.subheader("6. Diet Categories")
    
    diet_cols_all = DIET_HELPFUL_COLS + DIET_HARMFUL_COLS + DIET_NEUTRAL_COLS
    valid_diet_cols = [c for c in diet_cols_all if c in country_df.columns]
    
    if valid_diet_cols:
        diet_df = country_df[['year'] + valid_diet_cols].copy()
        
        # Z-score
        for c in valid_diet_cols:
            diet_df[c] = calc_zscore(diet_df[c])
            
        # Means
        if any(c in diet_df.columns for c in DIET_HELPFUL_COLS):
            diet_df['Helpful'] = diet_df[[c for c in DIET_HELPFUL_COLS if c in diet_df.columns]].mean(axis=1)
        if any(c in diet_df.columns for c in DIET_HARMFUL_COLS):
            diet_df['Harmful'] = diet_df[[c for c in DIET_HARMFUL_COLS if c in diet_df.columns]].mean(axis=1)
        if any(c in diet_df.columns for c in DIET_NEUTRAL_COLS):
            diet_df['Neutral'] = diet_df[[c for c in DIET_NEUTRAL_COLS if c in diet_df.columns]].mean(axis=1)
            
        diet_long = diet_df[['year', 'Helpful', 'Harmful', 'Neutral']].melt('year', var_name='category', value_name='value')
        
        fig6 = plot_with_trend(diet_long, 'year', 'value', 'category', DIET_PAL,
                               f"Diet Categories ({selected_iso})", "Z Score (Mean)")
        st.pyplot(fig6)

else:
    st.info("Please upload a CSV file to begin.")
