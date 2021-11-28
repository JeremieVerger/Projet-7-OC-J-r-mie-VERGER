# fichier permettant de déployer l'API via Streamlit

# mise en place de l'environnement python

import pandas as pd
import streamlit as st
import sys
sys.path.insert(0, 'API')
import pipeline

# chargement des données via le pipeline

@st.cache
def load_data():
    data_test = pipeline.predict('test')
    return data_test

# génération de l'API
data_test = load_data()
st.dataframe(data_test)