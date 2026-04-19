# Analiza Intarzierilor Zborurilor din SUA - 2015

Proiect **Pachete Software** | Poenariu Viorel, Popescu Andrei - Grupa 1093

Aplicatie Streamlit care analizeaza ~5.8 milioane de zboruri interne din SUA (2015) — intarzieri, anulari, rute, companii aeriene. Include: GeoPandas, tratare valori lipsa/extreme, codificare, scalare, statistici pandas, scikit-learn (K-Means, Regresie Logistica), statsmodels (regresie OLS).

**Set de date:** [Flight Delays - Kaggle](https://www.kaggle.com/datasets/usdot/flight-delays?resource=download). Fisierul `flights.csv` (~565MB) trebuie descarcat separat si pus in folder. Fara el, aplicatia genereaza date demonstrative.

**Rulare:** `pip install -r requirements.txt` apoi `python -m streamlit run flight_delays_app.py`
