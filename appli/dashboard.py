# voici le code permettant de générer le Dashboard via Streamlit

# mise en place de l'environnement Python

import pandas as pd
import streamlit as st
import sys
import pipeline
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

# on aura besoin d'une fonction pour supprimer les outliers. La voici
def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


# on créé la fonction qui va lire nos données

URL = 'https://share.streamlit.io/jeremieverger/projet-7-oc-j-r-mie-verger/main/appli/API.py'

@st.cache(allow_output_mutation=True)
def load_data():
    data_test = pd.read_json(URL)
    return data_test

# génération du dashboard

st.title("Prêt à dépenser")

st.text("Le chargement des données peut prendre quelques secondes.")

# on charge nos données
data_test = load_data()

# on demande à l'utilisateur de sélectionner l'identifiant d'un client
identifiant_client = st.selectbox("identifiant client",data_test['SK_ID_CURR'])

# affichage de la jauge de probabilité

jauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = int(data_test[data_test['SK_ID_CURR']==identifiant_client]["Probabilité"]*100),
    mode = "gauge+number",
    title = {'text': "Probabilité de défaut de paiement (en pourcentage)"},
    gauge = {'axis': {'range': [None, 100]},
             'bar': {'color': "darkgrey"},
             'bgcolor': "red",
             'steps' : [
                 {'range': [0, 35], 'color': "green"},
                 {'range': [35, 60], 'color': "orange"}],
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 50}}))

st.write(jauge)

# affichage des informations clients

st.header("Informations client")

colonnes = ["SK_ID_CURR","CODE_GENDER","AGE","CNT_CHILDREN","AMT_INCOME_TOTAL","NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","DAYS_EMPLOYED",
            "NAME_INCOME_TYPE","NAME_EDUCATION_TYPE","ORGANIZATION_TYPE",'AMT_CREDIT_ACTIVE']

var = st.selectbox("Variable ",colonnes)

info = data_test[data_test['SK_ID_CURR']==identifiant_client][var].values

text = var+' : '+str(info[0])
st.text(text)

# affichage d'un comparatif de nos résultats

st.header("Comparaison avec notre base de données:")

variables_num = ['AMT_INCOME_TOTAL','DAYS_EMPLOYED','AMT_CREDIT_ACTIVE','AGE']
variables_cat = ['ORGANIZATION_TYPE','CODE_GENDER',"NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","NAME_INCOME_TYPE",
                 "NAME_EDUCATION_TYPE"]

var_num = st.selectbox("Variable numérique",variables_num)

colors = ['green','red']
x = data_test[data_test["Prédiction"]==0][var_num]
x = x[~is_outlier(x)]
y = data_test[data_test["Prédiction"]==1][var_num]
y = y[~is_outlier(y)]

fig, ax = plt.subplots()
ax.hist([x, y], label=['Crédit accepté', 'Crédit refusé'], color=colors, density=True)
ax.set_title(var_num)
ax.legend(loc='upper right')
ax.axvline(x=float(data_test[data_test['SK_ID_CURR']==identifiant_client][var_num]),linewidth=5)

st.pyplot(fig)

var_cat = st.selectbox("Variable catégorielle",variables_cat)

x = data_test[data_test["Prédiction"]==0][var_cat]
y = data_test[data_test["Prédiction"]==1][var_cat]

fig = plt.figure(figsize = (10,4))
ax = fig.add_subplot(1,2,(1))
x.value_counts(normalize=True,dropna=True)[0:5].plot(kind='bar',title='Crédits acceptés', color = 'green')
ax = fig.add_subplot(1,2,(2))
y.value_counts(normalize=True,dropna=True)[0:5].plot(kind='bar',title='Crédits refusés', color = 'red')
fig.autofmt_xdate(rotation=45)

client = data_test[data_test['SK_ID_CURR']==identifiant_client][var_cat].values
text = var_cat+' : '+str(client[0])
st.text(text)
st.pyplot(fig)



