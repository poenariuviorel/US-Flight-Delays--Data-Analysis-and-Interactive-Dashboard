import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configurare pagina
st.set_page_config(
    page_title="Analiza Zborurilor SUA 2015",
    layout="wide"
)

st.title("Analiza Intarzierilor Zborurilor din SUA - 2015")
st.caption("Proiect Pachete Software | Set de date: Flight Delays (Kaggle)")

# ---------------------------------------------------------------
# Incarcare date
# ---------------------------------------------------------------
@st.cache_data
def load_airlines(path):
    return pd.read_csv(path)

@st.cache_data
def load_airports(path):
    return pd.read_csv(path)

@st.cache_data
def load_flights(path, nrows=None):
    df = pd.read_csv(path, nrows=nrows, low_memory=False)
    return df

# Sidebar - configurare
st.sidebar.header("Configurare")
airlines_path = st.sidebar.text_input("Cale airlines.csv", value="airlines.csv")
airports_path = st.sidebar.text_input("Cale airports.csv", value="airports.csv")
flights_path = st.sidebar.text_input("Cale flights.csv", value="flights.csv")

nrows = st.sidebar.slider("Cate randuri sa incarce din flights.csv:", 10000, 500000, 100000, step=10000)
st.sidebar.caption("Fisierul flights.csv are ~5.8 milioane randuri. Folosim un esantion pentru viteza.")

# Incarcare
try:
    airlines = load_airlines(airlines_path)
    airports = load_airports(airports_path)
except FileNotFoundError as e:
    st.error(f"Nu s-au gasit fisierele airlines/airports: {e}")
    st.stop()

try:
    flights = load_flights(flights_path, nrows=nrows)
except FileNotFoundError:
    st.warning("flights.csv nu a fost gasit. Se genereaza date demonstrative.")
    np.random.seed(42)
    n = 50000
    airline_codes = airlines['IATA_CODE'].tolist()
    airport_codes = airports['IATA_CODE'].dropna().tolist()[:50]

    flights = pd.DataFrame({
        'MONTH': np.random.randint(1, 13, n),
        'DAY': np.random.randint(1, 29, n),
        'DAY_OF_WEEK': np.random.randint(1, 8, n),
        'AIRLINE': np.random.choice(airline_codes, n),
        'FLIGHT_NUMBER': np.random.randint(100, 9999, n),
        'ORIGIN_AIRPORT': np.random.choice(airport_codes, n),
        'DESTINATION_AIRPORT': np.random.choice(airport_codes, n),
        'SCHEDULED_DEPARTURE': np.random.randint(500, 2300, n),
        'DEPARTURE_TIME': np.random.randint(500, 2359, n).astype(float),
        'DEPARTURE_DELAY': np.random.normal(5, 30, n).round(0),
        'TAXI_OUT': np.random.randint(5, 40, n),
        'WHEELS_OFF': np.random.randint(500, 2359, n),
        'SCHEDULED_TIME': np.random.randint(60, 400, n),
        'ELAPSED_TIME': np.random.randint(50, 420, n).astype(float),
        'AIR_TIME': np.random.randint(40, 380, n).astype(float),
        'DISTANCE': np.random.randint(100, 3000, n),
        'WHEELS_ON': np.random.randint(500, 2359, n),
        'TAXI_IN': np.random.randint(2, 25, n),
        'SCHEDULED_ARRIVAL': np.random.randint(500, 2359, n),
        'ARRIVAL_TIME': np.random.randint(500, 2359, n).astype(float),
        'ARRIVAL_DELAY': np.random.normal(3, 35, n).round(0),
        'DIVERTED': np.random.choice([0, 1], n, p=[0.98, 0.02]),
        'CANCELLED': np.random.choice([0, 1], n, p=[0.97, 0.03]),
        'CANCELLATION_REASON': np.random.choice([np.nan, 'A', 'B', 'C', 'D'], n, p=[0.97, 0.015, 0.01, 0.003, 0.002]),
        'AIR_SYSTEM_DELAY': np.zeros(n),
        'SECURITY_DELAY': np.zeros(n),
        'AIRLINE_DELAY': np.zeros(n),
        'LATE_AIRCRAFT_DELAY': np.zeros(n),
        'WEATHER_DELAY': np.zeros(n),
    })
    for col in ['DEPARTURE_TIME', 'ELAPSED_TIME', 'AIR_TIME', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']:
        mask = np.random.random(n) < 0.02
        flights.loc[mask, col] = np.nan

st.sidebar.markdown("---")
st.sidebar.write(f"Randuri incarcate: **{len(flights):,}**")

airline_names = airlines.set_index('IATA_CODE')['AIRLINE'].to_dict()

# ---------------------------------------------------------------
# Navigare
# ---------------------------------------------------------------
section = st.sidebar.selectbox(
    "Sectiune:",
    [
        "Prezentare generala",
        "Explorare date",
        "Valori lipsa si extreme",
        "Codificare date",
        "Scalare date",
        "Statistici si agregari",
        "Vizualizare geografica (GeoPandas)",
        "Machine Learning (Scikit-learn)",
        "Regresie multipla (Statsmodels)"
    ]
)

# ===============================================================
# PREZENTARE GENERALA
# ===============================================================
if section == "Prezentare generala":
    st.header("Prezentare generala")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total zboruri", f"{len(flights):,}")
    col2.metric("Companii aeriene", flights['AIRLINE'].nunique())
    col3.metric("Rata anulare", f"{flights['CANCELLED'].mean()*100:.1f}%")
    avg_d = flights['ARRIVAL_DELAY'].mean()
    col4.metric("Intarziere medie (min)", f"{avg_d:.1f}" if pd.notna(avg_d) else "N/A")

    st.markdown("---")

    st.subheader("Airlines")
    st.dataframe(airlines, use_container_width=True)

    st.subheader("Airports (primele 20)")
    st.dataframe(airports.head(20), use_container_width=True)

    st.subheader("Flights (primele 20)")
    st.dataframe(flights.head(20), use_container_width=True)

    st.subheader("Tipuri de date si valori lipsa - Flights")
    info_df = pd.DataFrame({
        'Coloana': flights.columns,
        'Tip': flights.dtypes.astype(str).values,
        'Non-null': flights.notna().sum().values,
        'Null': flights.isna().sum().values,
        'Null %': (flights.isna().mean() * 100).round(2).values
    })
    st.dataframe(info_df, use_container_width=True)

# ===============================================================
# EXPLORARE DATE
# ===============================================================
elif section == "Explorare date":
    st.header("Explorarea datelor")

    tab1, tab2, tab3 = st.tabs(["Distributii", "Corelatii", "Tendinte"])

    with tab1:
        st.subheader("Distributia intarzierilor la sosire")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        delay_data = flights['ARRIVAL_DELAY'].dropna().clip(-60, 120)
        axes[0].hist(delay_data, bins=80, color='steelblue', edgecolor='white', alpha=0.85)
        axes[0].set_title('ARRIVAL_DELAY (clip -60 la 120 min)')
        axes[0].set_xlabel('Intarziere (minute)')
        axes[0].set_ylabel('Frecventa')
        axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.6, label='La timp')
        axes[0].legend()

        top_al = flights['AIRLINE'].value_counts().head(6).index
        subset = flights[flights['AIRLINE'].isin(top_al)].copy()
        subset['delay_clip'] = subset['ARRIVAL_DELAY'].clip(-60, 120)
        subset['companie'] = subset['AIRLINE'].map(airline_names)
        sns.boxplot(data=subset, x='companie', y='delay_clip', ax=axes[1], palette='muted')
        axes[1].set_title('Intarzieri pe companie (top 6)')
        axes[1].set_xlabel('')
        axes[1].set_ylabel('Intarziere (min)')
        axes[1].tick_params(axis='x', rotation=25)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Distributia distantelor")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.hist(flights['DISTANCE'].dropna(), bins=60, color='seagreen', edgecolor='white', alpha=0.85)
        ax2.set_title('Distanta zborurilor (mile)')
        ax2.set_xlabel('Distanta')
        ax2.set_ylabel('Frecventa')
        st.pyplot(fig2)

    with tab2:
        st.subheader("Matrice de corelatii")
        corr_cols = [c for c in ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'DISTANCE',
                                  'SCHEDULED_TIME', 'ELAPSED_TIME', 'AIR_TIME',
                                  'TAXI_OUT', 'TAXI_IN', 'AIR_SYSTEM_DELAY',
                                  'AIRLINE_DELAY', 'WEATHER_DELAY', 'LATE_AIRCRAFT_DELAY']
                     if c in flights.columns]
        corr = flights[corr_cols].corr()
        fig3, ax3 = plt.subplots(figsize=(11, 7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=ax3, square=True, linewidths=0.5)
        ax3.set_title('Corelatii')
        plt.tight_layout()
        st.pyplot(fig3)

    with tab3:
        if 'MONTH' in flights.columns:
            st.subheader("Tendinte lunare")
            monthly = flights.groupby('MONTH').agg(
                nr_zboruri=('CANCELLED', 'count'),
                delay_mediu=('ARRIVAL_DELAY', 'mean'),
                rata_anulare=('CANCELLED', 'mean')
            ).reset_index()

            fig4, axes4 = plt.subplots(1, 3, figsize=(15, 4))
            axes4[0].bar(monthly['MONTH'], monthly['nr_zboruri'], color='steelblue')
            axes4[0].set_title('Zboruri pe luna')
            axes4[0].set_xlabel('Luna')

            axes4[1].plot(monthly['MONTH'], monthly['delay_mediu'], marker='o', color='indianred', linewidth=2)
            axes4[1].set_title('Intarziere medie pe luna')
            axes4[1].set_xlabel('Luna')
            axes4[1].axhline(y=0, color='gray', linestyle='--', alpha=0.4)

            axes4[2].bar(monthly['MONTH'], monthly['rata_anulare']*100, color='orange')
            axes4[2].set_title('Rata anulare (%)')
            axes4[2].set_xlabel('Luna')
            plt.tight_layout()
            st.pyplot(fig4)

        if 'DAY_OF_WEEK' in flights.columns:
            st.subheader("Intarzieri pe zi a saptamanii")
            day_names = {1:'Luni', 2:'Marti', 3:'Miercuri', 4:'Joi', 5:'Vineri', 6:'Sambata', 7:'Duminica'}
            weekly = flights.groupby('DAY_OF_WEEK')['ARRIVAL_DELAY'].mean().reset_index()
            weekly['zi'] = weekly['DAY_OF_WEEK'].map(day_names)
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            ax5.bar(weekly['zi'], weekly['ARRIVAL_DELAY'], color='slateblue')
            ax5.set_title('Intarziere medie pe zi')
            ax5.set_ylabel('Min')
            ax5.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig5)

# ===============================================================
# VALORI LIPSA SI EXTREME
# ===============================================================
elif section == "Valori lipsa si extreme":
    st.header("Tratarea valorilor lipsa si a outlier-ilor")

    tab1, tab2 = st.tabs(["Valori lipsa", "Outliers"])

    with tab1:
        st.subheader("Valori lipsa per coloana")
        missing = flights.isnull().sum()
        missing_pct = (flights.isnull().mean() * 100).round(2)
        missing_df = pd.DataFrame({
            'Coloana': missing.index,
            'Lipsa': missing.values,
            'Procent': missing_pct.values
        }).sort_values('Lipsa', ascending=False)
        missing_df = missing_df[missing_df['Lipsa'] > 0]

        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, max(3, len(missing_df)*0.35)))
            ax.barh(missing_df['Coloana'], missing_df['Procent'], color='indianred')
            ax.set_xlabel('Procent lipsa (%)')
            ax.set_title('Valori lipsa')
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.success("Nu exista valori lipsa.")

        st.markdown("---")
        st.subheader("Imputare valori lipsa")
        flights_clean = flights.copy()
        num_nan = flights_clean.select_dtypes(include=[np.number]).columns[
            flights_clean.select_dtypes(include=[np.number]).isnull().any()
        ].tolist()

        if num_nan:
            metoda = st.selectbox("Metoda de imputare (coloane numerice):",
                                  ["Medie", "Mediana", "Mod", "Eliminare randuri"])
            if st.button("Aplica"):
                before = len(flights_clean)
                if metoda == "Medie":
                    flights_clean[num_nan] = flights_clean[num_nan].fillna(flights_clean[num_nan].mean())
                elif metoda == "Mediana":
                    flights_clean[num_nan] = flights_clean[num_nan].fillna(flights_clean[num_nan].median())
                elif metoda == "Mod":
                    for c in num_nan:
                        m = flights_clean[c].mode()
                        if len(m) > 0:
                            flights_clean[c] = flights_clean[c].fillna(m[0])
                else:
                    flights_clean = flights_clean.dropna(subset=num_nan)
                after = len(flights_clean)
                st.write(f"Inainte: {before:,} randuri | Dupa: {after:,} randuri | "
                         f"Null ramase: {flights_clean[num_nan].isnull().sum().sum()}")

    with tab2:
        st.subheader("Detectare outliers (metoda IQR)")
        outlier_col = st.selectbox("Coloana:",
            [c for c in ['ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'DISTANCE', 'ELAPSED_TIME',
                         'AIR_TIME', 'TAXI_OUT', 'TAXI_IN'] if c in flights.columns])

        data = flights[outlier_col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = data[(data < lower) | (data > upper)]

        col1, col2, col3 = st.columns(3)
        col1.metric("Q1 / Q3", f"{Q1:.1f} / {Q3:.1f}")
        col2.metric("IQR", f"{IQR:.1f}")
        col3.metric("Outliers", f"{len(outliers):,} ({len(outliers)/len(data)*100:.1f}%)")
        st.write(f"Limite: [{lower:.1f}, {upper:.1f}]")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].boxplot(data, vert=True)
        axes[0].set_title(f'{outlier_col} - cu outliers')
        clean = data[(data >= lower) & (data <= upper)]
        axes[1].boxplot(clean, vert=True)
        axes[1].set_title(f'{outlier_col} - fara outliers')
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Metoda Z-Score")
        from scipy import stats
        z = np.abs(stats.zscore(data))
        prag = st.slider("Prag Z-Score:", 2.0, 4.0, 3.0, 0.5)
        z_out = (z > prag).sum()
        st.write(f"Outliers cu |Z| > {prag}: **{z_out:,}** ({z_out/len(data)*100:.1f}%)")

# ===============================================================
# CODIFICARE DATE
# ===============================================================
elif section == "Codificare date":
    st.header("Metode de codificare a datelor")

    from sklearn.preprocessing import LabelEncoder

    tab1, tab2, tab3 = st.tabs(["Label Encoding", "One-Hot Encoding", "Frequency Encoding"])

    with tab1:
        st.subheader("Label Encoding")
        st.write("Atribuie fiecarei categorii un numar. Util pentru variabile ordinale.")

        le = LabelEncoder()
        al_enc = pd.DataFrame({'Cod': flights['AIRLINE'].unique()})
        al_enc['Label'] = le.fit_transform(al_enc['Cod'])
        al_enc['Companie'] = al_enc['Cod'].map(airline_names)
        st.dataframe(al_enc.sort_values('Label'), use_container_width=True)

    with tab2:
        st.subheader("One-Hot Encoding")
        st.write("Creaza coloane binare (0/1) pentru fiecare categorie. Util pentru variabile nominale.")

        sample = flights[['AIRLINE']].head(1000)
        ohe = pd.get_dummies(sample, columns=['AIRLINE'], prefix='AL')
        st.write(f"Forma rezultat: {ohe.shape[0]} x {ohe.shape[1]}")
        st.dataframe(ohe.head(10), use_container_width=True)

    with tab3:
        st.subheader("Frequency Encoding")
        st.write("Inlocuieste categoria cu frecventa sa relativa.")

        if 'ORIGIN_AIRPORT' in flights.columns:
            freq = flights['ORIGIN_AIRPORT'].value_counts(normalize=True)
            freq_df = pd.DataFrame({'Aeroport': freq.head(15).index, 'Frecventa': freq.head(15).values.round(4)})
            st.dataframe(freq_df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(freq_df['Aeroport'], freq_df['Frecventa'], color='teal')
            ax.set_title('Frequency Encoding - top 15 aeroporturi origine')
            ax.set_ylabel('Frecventa relativa')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# ===============================================================
# SCALARE DATE
# ===============================================================
elif section == "Scalare date":
    st.header("Metode de scalare")

    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

    scale_cols = [c for c in ['DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'DISTANCE',
                               'ELAPSED_TIME', 'AIR_TIME', 'TAXI_OUT']
                  if c in flights.columns]
    data_s = flights[scale_cols].dropna().head(5000)

    tab1, tab2, tab3 = st.tabs(["StandardScaler", "MinMaxScaler", "RobustScaler"])

    with tab1:
        st.subheader("StandardScaler (media=0, std=1)")
        scaled = pd.DataFrame(StandardScaler().fit_transform(data_s), columns=scale_cols)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Inainte:")
            st.dataframe(data_s.describe().round(2), use_container_width=True)
        with col2:
            st.write("Dupa StandardScaler:")
            st.dataframe(scaled.describe().round(2), use_container_width=True)

    with tab2:
        st.subheader("MinMaxScaler (interval [0, 1])")
        mm = pd.DataFrame(MinMaxScaler().fit_transform(data_s), columns=scale_cols)
        st.dataframe(mm.describe().round(2), use_container_width=True)

    with tab3:
        st.subheader("RobustScaler (foloseste mediana si IQR)")
        rb = pd.DataFrame(RobustScaler().fit_transform(data_s), columns=scale_cols)
        st.dataframe(rb.describe().round(2), use_container_width=True)

    st.markdown("---")
    st.subheader("Comparatie vizuala")
    compare_col = st.selectbox("Coloana:", scale_cols)

    ss = pd.DataFrame(StandardScaler().fit_transform(data_s), columns=scale_cols)
    mm = pd.DataFrame(MinMaxScaler().fit_transform(data_s), columns=scale_cols)
    rb = pd.DataFrame(RobustScaler().fit_transform(data_s), columns=scale_cols)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, d, title in zip(axes, [data_s, ss, mm, rb],
                             ['Original', 'StandardScaler', 'MinMaxScaler', 'RobustScaler']):
        ax.hist(d[compare_col], bins=40, alpha=0.75, color='steelblue', edgecolor='white')
        ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)

# ===============================================================
# STATISTICI SI AGREGARI
# ===============================================================
elif section == "Statistici si agregari":
    st.header("Prelucrari statistice, grupare si agregare")

    tab1, tab2, tab3 = st.tabs(["Statistici descriptive", "Grupare si agregare", "Functii de grup"])

    with tab1:
        st.subheader("Statistici descriptive")
        st.dataframe(flights.describe().round(2), use_container_width=True)

        st.subheader("Per companie aeriana")
        al_stats = flights.groupby('AIRLINE').agg(
            nr_zboruri=('AIRLINE', 'count'),
            delay_mediu=('ARRIVAL_DELAY', 'mean'),
            delay_median=('ARRIVAL_DELAY', 'median'),
            delay_std=('ARRIVAL_DELAY', 'std'),
            distanta_medie=('DISTANCE', 'mean'),
            rata_anulare=('CANCELLED', 'mean')
        ).round(2)
        al_stats['Companie'] = al_stats.index.map(airline_names)
        al_stats['rata_anulare'] = (al_stats['rata_anulare'] * 100).round(2)
        st.dataframe(al_stats.sort_values('delay_mediu'), use_container_width=True)

    with tab2:
        st.subheader("Pivot table: intarziere medie (luna x zi saptamana)")
        if 'MONTH' in flights.columns and 'DAY_OF_WEEK' in flights.columns:
            pivot = flights.pivot_table(values='ARRIVAL_DELAY', index='MONTH',
                                        columns='DAY_OF_WEEK', aggfunc='mean').round(1)
            day_names = {1:'Lun', 2:'Mar', 3:'Mie', 4:'Joi', 5:'Vin', 6:'Sam', 7:'Dum'}
            pivot.columns = [day_names.get(c, c) for c in pivot.columns]
            month_names = {1:'Ian', 2:'Feb', 3:'Mar', 4:'Apr', 5:'Mai', 6:'Iun',
                          7:'Iul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
            pivot.index = [month_names.get(i, i) for i in pivot.index]

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', center=0, ax=ax, linewidths=0.5)
            ax.set_title('Intarziere medie la sosire (minute)')
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("Grupare pe categorii de distanta")
        ft = flights.copy()
        ft['categorie_dist'] = pd.cut(ft['DISTANCE'],
            bins=[0, 500, 1000, 2000, 5000],
            labels=['Scurt (<500mi)', 'Mediu (500-1000)', 'Lung (1000-2000)', 'F. lung (>2000)'])
        dg = ft.groupby('categorie_dist', observed=True).agg(
            zboruri=('DISTANCE', 'count'),
            delay_mediu=('ARRIVAL_DELAY', 'mean'),
            dist_medie=('DISTANCE', 'mean')
        ).round(2)
        st.dataframe(dg, use_container_width=True)

    with tab3:
        st.subheader("Top 15 rute")
        if 'ORIGIN_AIRPORT' in flights.columns:
            ft2 = flights.copy()
            ft2['ruta'] = ft2['ORIGIN_AIRPORT'] + ' -> ' + ft2['DESTINATION_AIRPORT']
            top_r = ft2.groupby('ruta').agg(
                zboruri=('ruta', 'count'),
                delay_mediu=('ARRIVAL_DELAY', 'mean'),
                anulare=('CANCELLED', 'mean')
            ).sort_values('zboruri', ascending=False).head(15).round(2)
            top_r['anulare'] = (top_r['anulare'] * 100).round(2)
            st.dataframe(top_r, use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(top_r.index[::-1], top_r['zboruri'][::-1], color='steelblue')
            ax.set_title('Top 15 rute - numar de zboruri')
            ax.set_xlabel('Zboruri')
            plt.tight_layout()
            st.pyplot(fig)

        st.subheader("Agregare multipla pe ARRIVAL_DELAY")
        multi = flights.groupby('AIRLINE')['ARRIVAL_DELAY'].agg(
            ['count', 'mean', 'median', 'std', 'min', 'max']
        ).round(2)
        multi.index = multi.index.map(airline_names)
        st.dataframe(multi, use_container_width=True)

# ===============================================================
# GEOPANDAS
# ===============================================================
elif section == "Vizualizare geografica (GeoPandas)":
    st.header("Vizualizare geografica cu GeoPandas")

    try:
        import geopandas as gpd
        from shapely.geometry import Point, LineString

        airports_clean = airports.dropna(subset=['LATITUDE', 'LONGITUDE'])
        geometry = [Point(lon, lat) for lon, lat in zip(airports_clean['LONGITUDE'], airports_clean['LATITUDE'])]
        gdf = gpd.GeoDataFrame(airports_clean, geometry=geometry, crs='EPSG:4326')

        # Incarcam harta lumii (compatibil cu geopandas >= 1.0)
        @st.cache_data
        def load_us_map():
            url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
            try:
                world = gpd.read_file(url)
            except Exception:
                return None
            us = world[world['ISO_A3'] == 'USA']
            return us

        us = load_us_map()

        tab1, tab2, tab3 = st.tabs(["Harta aeroporturi", "Rute populare", "Intarzieri pe harta"])

        with tab1:
            st.subheader("Aeroporturile din SUA")

            fig, ax = plt.subplots(figsize=(13, 7))
            if us is not None:
                us.plot(ax=ax, color='#f0f0f0', edgecolor='#bbb')

            if 'ORIGIN_AIRPORT' in flights.columns:
                counts = flights['ORIGIN_AIRPORT'].value_counts()
                gdf['zboruri'] = gdf['IATA_CODE'].map(counts).fillna(0)
                gdf_us = gdf[(gdf['LONGITUDE'] > -130) & (gdf['LONGITUDE'] < -65) &
                             (gdf['LATITUDE'] > 24) & (gdf['LATITUDE'] < 50)]

                gdf_us.plot(ax=ax, markersize=gdf_us['zboruri']/gdf_us['zboruri'].max()*150 + 3,
                           color='indianred', alpha=0.6, edgecolor='darkred', linewidth=0.3)

                top10 = gdf_us.nlargest(10, 'zboruri')
                for _, r in top10.iterrows():
                    ax.annotate(r['IATA_CODE'], xy=(r['LONGITUDE'], r['LATITUDE']),
                              fontsize=7, fontweight='bold', ha='center', va='bottom')

            ax.set_xlim(-130, -65)
            ax.set_ylim(24, 50)
            ax.set_title('Aeroporturi SUA (dimensiune = volum trafic)')
            plt.tight_layout()
            st.pyplot(fig)

        with tab2:
            st.subheader("Rute aeriene pe harta")
            if 'ORIGIN_AIRPORT' in flights.columns:
                ft = flights.copy()
                ft['ruta'] = ft['ORIGIN_AIRPORT'] + '-' + ft['DESTINATION_AIRPORT']
                top_n = st.slider("Numar de rute:", 5, 25, 10)
                top_routes = ft['ruta'].value_counts().head(top_n)

                coords = airports_clean.set_index('IATA_CODE')[['LATITUDE', 'LONGITUDE']].to_dict('index')

                fig2, ax2 = plt.subplots(figsize=(13, 7))
                if us is not None:
                    us.plot(ax=ax2, color='#f5f5f5', edgecolor='#ccc')

                colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(top_routes)))
                for idx, (route, count) in enumerate(top_routes.items()):
                    o, d = route.split('-')
                    if o in coords and d in coords:
                        oc, dc = coords[o], coords[d]
                        line = LineString([(oc['LONGITUDE'], oc['LATITUDE']),
                                          (dc['LONGITUDE'], dc['LATITUDE'])])
                        gpd.GeoDataFrame(geometry=[line], crs='EPSG:4326').plot(
                            ax=ax2, color=colors[idx],
                            linewidth=count/top_routes.max()*3 + 0.5, alpha=0.7)

                ax2.set_xlim(-130, -65)
                ax2.set_ylim(24, 50)
                ax2.set_title(f'Top {top_n} rute')
                plt.tight_layout()
                st.pyplot(fig2)

        with tab3:
            st.subheader("Intarziere medie per aeroport")
            if 'ORIGIN_AIRPORT' in flights.columns:
                delay_ap = flights.groupby('ORIGIN_AIRPORT')['ARRIVAL_DELAY'].mean().reset_index()
                delay_ap.columns = ['IATA_CODE', 'avg_delay']
                gdf_d = gdf.merge(delay_ap, on='IATA_CODE', how='inner')
                gdf_d = gdf_d[(gdf_d['LONGITUDE'] > -130) & (gdf_d['LONGITUDE'] < -65) &
                               (gdf_d['LATITUDE'] > 24) & (gdf_d['LATITUDE'] < 50)]

                fig3, ax3 = plt.subplots(figsize=(13, 7))
                if us is not None:
                    us.plot(ax=ax3, color='#f5f5f5', edgecolor='#ccc')
                gdf_d.plot(ax=ax3, column='avg_delay', cmap='RdYlGn_r', markersize=25,
                          alpha=0.7, legend=True,
                          legend_kwds={'label': 'Intarziere medie (min)', 'shrink': 0.5})
                ax3.set_xlim(-130, -65)
                ax3.set_ylim(24, 50)
                ax3.set_title('Intarziere medie pe aeroport')
                plt.tight_layout()
                st.pyplot(fig3)

    except ImportError:
        st.error("geopandas nu este instalat. Rulati: pip install geopandas")

# ===============================================================
# MACHINE LEARNING
# ===============================================================
elif section == "Machine Learning (Scikit-learn)":
    st.header("Machine Learning cu Scikit-learn")

    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
    from sklearn.preprocessing import StandardScaler

    tab1, tab2 = st.tabs(["Clusterizare K-Means", "Regresie Logistica"])

    with tab1:
        st.subheader("K-Means Clustering")
        st.write("Grupam zborurile dupa distanta, intarziere si timp de zbor.")

        cl_cols = [c for c in ['DISTANCE', 'ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'AIR_TIME', 'ELAPSED_TIME']
                   if c in flights.columns]
        cl_data = flights[cl_cols].dropna()
        if len(cl_data) > 10000:
            cl_data = cl_data.sample(10000, random_state=42)

        scaler = StandardScaler()
        cl_scaled = scaler.fit_transform(cl_data)

        # Elbow + Silhouette
        ks = range(2, 9)
        inertias, sils = [], []
        for k in ks:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(cl_scaled)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(cl_scaled, km.labels_, sample_size=2000))

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(list(ks), inertias, 'o-', color='steelblue')
        axes[0].set_title('Metoda Elbow')
        axes[0].set_xlabel('K')
        axes[0].set_ylabel('Inertie')
        axes[1].plot(list(ks), sils, 'o-', color='indianred')
        axes[1].set_title('Silhouette Score')
        axes[1].set_xlabel('K')
        plt.tight_layout()
        st.pyplot(fig)

        n_k = st.slider("Numar de clustere:", 2, 8, 3)
        km_f = KMeans(n_clusters=n_k, random_state=42, n_init=10)
        cl_data = cl_data.copy()
        cl_data['Cluster'] = km_f.fit_predict(cl_scaled)

        st.write(f"Silhouette Score: **{silhouette_score(cl_scaled, cl_data['Cluster'], sample_size=2000):.3f}**")

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        scatter = ax2.scatter(cl_data['DISTANCE'], cl_data['ARRIVAL_DELAY'].clip(-100, 200),
                             c=cl_data['Cluster'], cmap='viridis', alpha=0.5, s=8)
        ax2.set_xlabel('Distanta (mile)')
        ax2.set_ylabel('Intarziere sosire (min)')
        ax2.set_title(f'K-Means (K={n_k})')
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        st.pyplot(fig2)

        st.write("Statistici per cluster:")
        st.dataframe(cl_data.groupby('Cluster')[cl_cols].mean().round(2), use_container_width=True)

    with tab2:
        st.subheader("Regresie Logistica - clasificare intarziat/la timp")
        st.write("Un zbor e considerat intarziat daca ARRIVAL_DELAY > 15 min.")

        lr_cols = [c for c in ['DEPARTURE_DELAY', 'DISTANCE', 'SCHEDULED_TIME', 'TAXI_OUT', 'MONTH', 'DAY_OF_WEEK']
                  if c in flights.columns]
        lr_data = flights[lr_cols + ['ARRIVAL_DELAY']].dropna()
        lr_data['delayed'] = (lr_data['ARRIVAL_DELAY'] > 15).astype(int)

        if len(lr_data) > 50000:
            lr_data = lr_data.sample(50000, random_state=42)

        X = lr_data[lr_cols]
        y = lr_data['delayed']
        st.write(f"La timp: {(y==0).sum():,} | Intarziat: {(y==1).sum():,}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        sc = StandardScaler()
        X_train_s = sc.fit_transform(X_train)
        X_test_s = sc.transform(X_test)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train_s, y_train)
        y_pred = lr.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)

        st.metric("Acuratete", f"{acc:.2%}")

        cm = confusion_matrix(y_test, y_pred)
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes3[0],
                   xticklabels=['La timp', 'Intarziat'], yticklabels=['La timp', 'Intarziat'])
        axes3[0].set_title('Matrice de confuzie')
        axes3[0].set_xlabel('Predictie')
        axes3[0].set_ylabel('Real')

        coef_df = pd.DataFrame({'Feature': lr_cols, 'Coeficient': lr.coef_[0]}).sort_values('Coeficient', key=abs)
        axes3[1].barh(coef_df['Feature'], coef_df['Coeficient'],
                     color=['indianred' if c > 0 else 'seagreen' for c in coef_df['Coeficient']])
        axes3[1].set_title('Importanta feature-urilor')
        plt.tight_layout()
        st.pyplot(fig3)

        st.subheader("Raport de clasificare")
        report = classification_report(y_test, y_pred, target_names=['La timp', 'Intarziat'], output_dict=True)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ===============================================================
# REGRESIE MULTIPLA
# ===============================================================
elif section == "Regresie multipla (Statsmodels)":
    st.header("Regresie multipla cu Statsmodels")

    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    st.write("Model de regresie liniara multipla pentru prezicerea ARRIVAL_DELAY.")

    reg_cols = [c for c in ['DEPARTURE_DELAY', 'DISTANCE', 'SCHEDULED_TIME', 'TAXI_OUT', 'TAXI_IN']
               if c in flights.columns]
    reg_data = flights[reg_cols + ['ARRIVAL_DELAY']].dropna()
    if len(reg_data) > 30000:
        reg_data = reg_data.sample(30000, random_state=42)

    selected = st.multiselect("Variabile predictoare:", reg_cols, default=reg_cols)

    if len(selected) > 0:
        X = reg_data[selected]
        y = reg_data['ARRIVAL_DELAY']
        X_c = sm.add_constant(X)

        model = sm.OLS(y, X_c).fit()

        st.subheader("Summary OLS")
        st.text(model.summary().as_text())

        col1, col2, col3 = st.columns(3)
        col1.metric("R-squared", f"{model.rsquared:.4f}")
        col2.metric("R-squared adj.", f"{model.rsquared_adj:.4f}")
        col3.metric("F-statistic", f"{model.fvalue:.2f}")

        st.subheader("Coeficienti")
        coef = pd.DataFrame({
            'Variabila': model.params.index,
            'Coeficient': model.params.values.round(4),
            'Std Error': model.bse.values.round(4),
            't': model.tvalues.values.round(4),
            'p-value': model.pvalues.values.round(6)
        })
        st.dataframe(coef, use_container_width=True)

        if len(selected) > 1:
            st.subheader("Multicoliniaritate (VIF)")
            vif = pd.DataFrame()
            vif['Feature'] = selected
            vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(selected))]
            st.dataframe(vif.round(2), use_container_width=True)
            st.write("VIF < 5: ok | VIF 5-10: moderat | VIF > 10: problematic")

        st.subheader("Diagnostice")
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        res = model.resid
        fitted = model.fittedvalues

        axes[0, 0].scatter(fitted, res, alpha=0.15, s=4, color='steelblue')
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_title('Reziduuri vs Valori estimate')
        axes[0, 0].set_xlabel('Estimate')
        axes[0, 0].set_ylabel('Reziduuri')

        from scipy import stats
        stats.probplot(res, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')

        axes[1, 0].hist(res, bins=60, color='steelblue', edgecolor='white', alpha=0.8, density=True)
        axes[1, 0].set_title('Distributia reziduurilor')

        idx = np.random.choice(len(y), min(2000, len(y)), replace=False)
        axes[1, 1].scatter(y.iloc[idx], fitted.iloc[idx], alpha=0.15, s=4, color='slateblue')
        axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
        axes[1, 1].set_title('Real vs Estimat')
        axes[1, 1].set_xlabel('Real')
        axes[1, 1].set_ylabel('Estimat')

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Selectati cel putin o variabila.")

# ---------------------------------------------------------------
# Sidebar footer
# ---------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.write("Facilitati: Streamlit, GeoPandas, valori lipsa/extreme, "
                 "codificare, scalare, statistici pandas, functii de grup, "
                 "scikit-learn (K-Means + Regresie Logistica), statsmodels (OLS)")
st.sidebar.caption("Proiect Pachete Software 2025")