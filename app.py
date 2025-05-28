
import streamlit as st
import pandas as pd
import joblib

# 🔃 Modell laden
model = joblib.load('random_forest_houseprice_model.pkl')

st.set_page_config(page_title="Hauspreis-Schätzer", layout="centered")
st.title("🏡 KI-gestützte Hauspreis-Vorhersage")
st.caption("Ein Prototyp für Zillow – erstellt vom Data Engineer")

# 🔢 Eingabefelder
st.subheader("📋 Gebe die Eckdaten zum Haus ein:")

OverallQual = st.slider("Bauqualität (OverallQual)", 1, 10, 5)
GrLivArea = st.number_input("Wohnfläche (GrLivArea) in sqft", 500, 5000, 1500)
GarageCars = st.slider("Anzahl Garagenplätze", 0, 4, 2)
TotalBsmtSF = st.number_input("Kellerfläche (TotalBsmtSF) in sqft", 0, 3000, 800)
FullBath = st.slider("Anzahl Vollbäder", 0, 4, 2)
YearBuilt = st.number_input("Baujahr", 1900, 2024, 2000)
YearRemodAdd = st.number_input("Renovierungsjahr", 1900, 2024, 2005)
Neighborhood = st.selectbox("Lage (Neighborhood)", ['CollgCr', 'NridgHt', 'Somerst', 'OldTown'])
KitchenQual = st.selectbox("Küchenqualität", ['Ex', 'Gd', 'TA', 'Fa'])
ExterQual = st.selectbox("Außenqualität", ['Ex', 'Gd', 'TA', 'Fa'])

# 📦 Eingabedaten als DataFrame
input_data = pd.DataFrame({
    'OverallQual': [OverallQual],
    'GrLivArea': [GrLivArea],
    'GarageCars': [GarageCars],
    'TotalBsmtSF': [TotalBsmtSF],
    'FullBath': [FullBath],
    'YearBuilt': [YearBuilt],
    'YearRemodAdd': [YearRemodAdd],
    'Neighborhood': [Neighborhood],
    'KitchenQual': [KitchenQual],
    'ExterQual': [ExterQual]
})

# 🧠 One-Hot-Encoding
input_encoded = pd.get_dummies(input_data)
for col in model.feature_names_in_:
    if col not in input_encoded:
        input_encoded[col] = 0
input_encoded = input_encoded[model.feature_names_in_]

# 🧾 Vorhersage
predicted_price = model.predict(input_encoded)[0]

# 💬 Ausgabe
st.subheader("💰 Geschätzter Verkaufspreis:")
st.success(f"{predicted_price:,.2f} $")
