# Voici le pipeline permettant de traiter les données brutes dans l'optique d'en déduire la probabilité de défaut de paiement.

# mise en place de l'environnement Python

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import manifold, decomposition
from sklearn import preprocessing
import joblib
from sklearn import linear_model
import zipfile as zf
import urllib.request

# on récupère puis décompresse les fichiers CSV contenant les données
urllib.request.urlretrieve('https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip', "Data.zip")
files = zf.ZipFile("Data.zip",'r')
files.extractall("P7_data")
files.close()

# création d'une fonction de détection des valeurs manquantes

def is_nan(x):
    return (x != x)

# Ces premières fonctions importent les données brutes et en fond une première sélection de variables

def application_test():
    appli_test = pd.read_csv('./P7_data/application_test.csv', sep = ',')
    #on sélectionne les variables pertinentes
    colonnes_test = ["SK_ID_CURR","NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY",
                "CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
                "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","DAYS_BIRTH","DAYS_EMPLOYED",
                "CNT_FAM_MEMBERS","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
                "OWN_CAR_AGE","REGION_RATING_CLIENT_W_CITY","ORGANIZATION_TYPE"]
    appli_test = appli_test[colonnes_test]
    return appli_test

def application_train():
    appli_train = pd.read_csv('./P7_data/application_train.csv', sep = ',')
     #on sélectionne les variables pertinentes
    colonnes_train = ["SK_ID_CURR","NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY",
                "CNT_CHILDREN","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
                "NAME_FAMILY_STATUS","NAME_HOUSING_TYPE","DAYS_BIRTH","DAYS_EMPLOYED",
                "CNT_FAM_MEMBERS","NAME_INCOME_TYPE","NAME_EDUCATION_TYPE",
                "OWN_CAR_AGE","REGION_RATING_CLIENT_W_CITY","ORGANIZATION_TYPE","TARGET"]
    appli_train = appli_train[colonnes_train]
    return appli_train

def bureau():
    bureau = pd.read_csv('./P7_data/bureau.csv', sep = ',')
    # sélection des variables pertinentes
    colonnes = ["SK_ID_CURR","SK_ID_BUREAU","CREDIT_ACTIVE","CREDIT_DAY_OVERDUE","DAYS_CREDIT_ENDDATE",
           "AMT_CREDIT_MAX_OVERDUE","AMT_CREDIT_SUM_DEBT","AMT_CREDIT_SUM_OVERDUE",
           "CREDIT_TYPE"]
    bureau = bureau[colonnes]
    return bureau

def credit_card():
    credit_card = pd.read_csv('./P7_data/credit_card_balance.csv', sep = ',')
    # sélection des variables pertinentes 
    colonnes = ["SK_ID_PREV","SK_ID_CURR","AMT_BALANCE","AMT_DRAWINGS_CURRENT"]
    credit_card = credit_card[colonnes]
    credit_card = credit_card.groupby(credit_card['SK_ID_PREV'], as_index = False).agg(SK_ID_CURR=('SK_ID_CURR', 'first'),
                                                                                   AMT_BALANCE_mean=('AMT_BALANCE','mean'),
                                                                                   AMT_BALANCE_min=('AMT_BALANCE',min),
                                                                                   AMT_BALANCE_max=('AMT_BALANCE',max),
                                                                                   AMT_DRAWINGS_CURRENT=('AMT_DRAWINGS_CURRENT','mean'))
    return credit_card

def installments():
    installments = pd.read_csv('./P7_data/installments_payments.csv', sep = ',')
    # sélection des variables pertinentes
    colonnes = ["SK_ID_PREV","SK_ID_CURR","AMT_INSTALMENT","AMT_PAYMENT"]
    installments = installments[colonnes]
    installments = installments.groupby(installments['SK_ID_PREV'], as_index = False).agg(SK_ID_CURR=('SK_ID_CURR', 'first'),
                                                                                      AMT_INSTALMENT=('AMT_INSTALMENT',sum),
                                                                                      AMT_PAYMENT=('AMT_PAYMENT',sum))
    return installments

def POS():
    POS = pd.read_csv('./P7_data/POS_CASH_balance.csv', sep = ',')
    # sélection des variables pertinentes
    colonnes = ["SK_ID_PREV","SK_ID_CURR","CNT_INSTALMENT_FUTURE"]
    POS = POS[colonnes]
    # afin de gagner de la mémoire, on agglomère pour chaque prêt ce qu'il reste à payer 
    #(ie : le minimum de la variable CNT_INSTALMENT_FUTURE)
    POS = POS.groupby(POS['SK_ID_PREV'], as_index = False).agg( SK_ID_CURR=('SK_ID_CURR', 'first'), 
                                                           CNT_INSTALMENT_FUTURE=('CNT_INSTALMENT_FUTURE',min))
    return POS

def previous():
    previous_appli = pd.read_csv('./P7_data/previous_application.csv', sep = ',')
    # sélection des variables pertinentes
    colonnes = ["SK_ID_PREV","SK_ID_CURR","NAME_CONTRACT_TYPE","AMT_CREDIT","NAME_CONTRACT_STATUS",
           "CODE_REJECT_REASON"]
    previous_appli = previous_appli[colonnes]
    return previous_appli

# on agglomère les différents Dataframe ensemble

def agglomération():
    # on rassemble les données application et bureau
    data_test = pd.merge(application_test(), bureau(),how = "left", on = "SK_ID_CURR")
    data_train = pd.merge(application_train(), bureau(),how = "left", on = "SK_ID_CURR")
    # on ajoute les données previous_appli
    data_test = pd.merge(data_test, previous(),how = "left", on = "SK_ID_CURR")
    data_train = pd.merge(data_train, previous(),how = "left", on = "SK_ID_CURR")
    # on ajoute les données POS
    data_test = pd.merge(data_test, POS(),how = "left", on = ["SK_ID_PREV","SK_ID_CURR"])
    data_train = pd.merge(data_train, POS(),how = "left", on = ["SK_ID_PREV","SK_ID_CURR"])
    # on ajoute les données installments
    data_test = pd.merge(data_test, installments(),how = "left", on = ["SK_ID_PREV","SK_ID_CURR"])
    data_train = pd.merge(data_train, installments(),how = "left", on = ["SK_ID_PREV","SK_ID_CURR"])
    # enfin, on ajoute les données credit card
    data_test = pd.merge(data_test, credit_card(),how = "left", on = ["SK_ID_PREV","SK_ID_CURR"])
    data_train = pd.merge(data_train, credit_card(),how = "left", on = ["SK_ID_PREV","SK_ID_CURR"])
    
    # on renomme les variables afin qu'il n'y ait pas d'ambiguïté
    names = {'NAME_CONTRACT_TYPE_x':'NAME_CONTRACT_TYPE_CURR',
        'NAME_CONTRACT_TYPE_y':'NAME_CONTRACT_TYPE_PREV',
        'AMT_CREDIT_x':'AMT_CREDIT_CURR',
        'AMT_CREDIT_y':'AMT_CREDIT_PREV'}
    data_train.rename(columns = names, inplace = True)
    data_test.rename(columns = names, inplace = True)
    
    # on va maintenant rassembler les lignes entre elles pour que chaque ligne corresponde à une demande de crédit actuelle
    data_train = data_train.groupby(data_train['SK_ID_CURR'], as_index = False).agg(NAME_CONTRACT_TYPE_CURR=('NAME_CONTRACT_TYPE_CURR',
                                                                                                             'first'),
                                                                                CODE_GENDER=('CODE_GENDER','first'),
                                                                                FLAG_OWN_CAR=('FLAG_OWN_CAR','first'),
                                                                                FLAG_OWN_REALTY=('FLAG_OWN_REALTY','first'),
                                                                                CNT_CHILDREN=('CNT_CHILDREN','first'),
                                                                                AMT_INCOME_TOTAL=('AMT_INCOME_TOTAL','first'),
                                                                                AMT_CREDIT_CURR=('AMT_CREDIT_CURR','first'),
                                                                                AMT_ANNUITY=('AMT_ANNUITY','first'),
                                                                                NAME_FAMILY_STATUS=('NAME_FAMILY_STATUS','first'),
                                                                                NAME_HOUSING_TYPE=('NAME_HOUSING_TYPE','first'),
                                                                                DAYS_BIRTH=('DAYS_BIRTH','first'),
                                                                                DAYS_EMPLOYED=('DAYS_EMPLOYED','first'),
                                                                                CNT_FAM_MEMBERS=('CNT_FAM_MEMBERS','first'),
                                                                                NAME_INCOME_TYPE=('NAME_INCOME_TYPE','first'),
                                                                                NAME_EDUCATION_TYPE=('NAME_EDUCATION_TYPE','first'),
                                                                                OWN_CAR_AGE=("OWN_CAR_AGE",'first'),
                                                                                REGION_RATING_CLIENT_W_CITy=
                                                                                    ('REGION_RATING_CLIENT_W_CITY','first'),
                                                                                ORGANIZATION_TYPE=("ORGANIZATION_TYPE","first"),
                                                                                TARGET=('TARGET','first'),
                                                                                SK_ID_BUREAU=('SK_ID_BUREAU','first'),
                                                                                CREDIT_ACTIVE=('CREDIT_ACTIVE',list),
                                                                                CREDIT_DAY_OVERDUE=('CREDIT_DAY_OVERDUE','max'),
                                                                                DAYS_CREDIT_ENDDATE=('DAYS_CREDIT_ENDDATE','max'),
                                                                                AMT_CREDIT_MAX_OVERDUE=('AMT_CREDIT_MAX_OVERDUE','max'),
                                                                                AMT_CREDIT_SUM_DEBT=('AMT_CREDIT_SUM_DEBT',sum),
                                                                                CREDIT_TYPE=('CREDIT_TYPE',set),
                                                                                SK_ID_PREV=('SK_ID_PREV','first'),
                                                                                NAME_CONTRACT_TYPE_PREV=('NAME_CONTRACT_TYPE_PREV','first'),
                                                                                AMT_CREDIT_PREV=('AMT_CREDIT_PREV','first'),
                                                                                NAME_CONTRACT_STATUS=('NAME_CONTRACT_STATUS','first'),
                                                                                CODE_REJECT_REASON=('CODE_REJECT_REASON','first'),
                                                                                CNT_INSTALMENT_FUTURE=('CNT_INSTALMENT_FUTURE',set),
                                                                                AMT_INSTALMENT=('AMT_INSTALMENT',set),
                                                                                AMT_PAYMENT=('AMT_PAYMENT',set),
                                                                                AMT_BALANCE_mean=('AMT_BALANCE_mean','mean'),
                                                                                AMT_BALANCE_min=('AMT_BALANCE_min',min),
                                                                                AMT_BALANCE_max=('AMT_BALANCE_max',max),
                                                                                AMT_DRAWINGS_CURRENT=('AMT_DRAWINGS_CURRENT','mean'))
    
    data_test = data_test.groupby(data_test['SK_ID_CURR'], as_index = False).agg(NAME_CONTRACT_TYPE_CURR=('NAME_CONTRACT_TYPE_CURR',
                                                                                                          'first'),
                                                                                CODE_GENDER=('CODE_GENDER','first'),
                                                                                FLAG_OWN_CAR=('FLAG_OWN_CAR','first'),
                                                                                FLAG_OWN_REALTY=('FLAG_OWN_REALTY','first'),
                                                                                CNT_CHILDREN=('CNT_CHILDREN','first'),
                                                                                AMT_INCOME_TOTAL=('AMT_INCOME_TOTAL','first'),
                                                                                AMT_CREDIT_CURR=('AMT_CREDIT_CURR','first'),
                                                                                AMT_ANNUITY=('AMT_ANNUITY','first'),
                                                                                NAME_FAMILY_STATUS=('NAME_FAMILY_STATUS','first'),
                                                                                NAME_HOUSING_TYPE=('NAME_HOUSING_TYPE','first'),
                                                                                DAYS_BIRTH=('DAYS_BIRTH','first'),
                                                                                DAYS_EMPLOYED=('DAYS_EMPLOYED','first'),
                                                                                CNT_FAM_MEMBERS=('CNT_FAM_MEMBERS','first'),
                                                                                NAME_INCOME_TYPE=('NAME_INCOME_TYPE','first'),
                                                                                NAME_EDUCATION_TYPE=('NAME_EDUCATION_TYPE','first'),
                                                                                OWN_CAR_AGE=("OWN_CAR_AGE",'first'),
                                                                                REGION_RATING_CLIENT_W_CITY=
                                                                                 ('REGION_RATING_CLIENT_W_CITY','first'),
                                                                                ORGANIZATION_TYPE=("ORGANIZATION_TYPE","first"),
                                                                                SK_ID_BUREAU=('SK_ID_BUREAU','first'),
                                                                                CREDIT_ACTIVE=('CREDIT_ACTIVE',list),
                                                                                CREDIT_DAY_OVERDUE=('CREDIT_DAY_OVERDUE','max'),
                                                                                DAYS_CREDIT_ENDDATE=('DAYS_CREDIT_ENDDATE','max'),
                                                                                AMT_CREDIT_MAX_OVERDUE=('AMT_CREDIT_MAX_OVERDUE','max'),
                                                                                AMT_CREDIT_SUM_DEBT=('AMT_CREDIT_SUM_DEBT',sum),
                                                                                CREDIT_TYPE=('CREDIT_TYPE',set),
                                                                                SK_ID_PREV=('SK_ID_PREV','first'),
                                                                                NAME_CONTRACT_TYPE_PREV=('NAME_CONTRACT_TYPE_PREV','first'),
                                                                                AMT_CREDIT_PREV=('AMT_CREDIT_PREV','first'),
                                                                                NAME_CONTRACT_STATUS=('NAME_CONTRACT_STATUS','first'),
                                                                                CODE_REJECT_REASON=('CODE_REJECT_REASON','first'),
                                                                                CNT_INSTALMENT_FUTURE=('CNT_INSTALMENT_FUTURE',set),
                                                                                AMT_INSTALMENT=('AMT_INSTALMENT',set),
                                                                                AMT_PAYMENT=('AMT_PAYMENT',set),
                                                                                AMT_BALANCE_mean=('AMT_BALANCE_mean','mean'),
                                                                                AMT_BALANCE_min=('AMT_BALANCE_min',min),
                                                                                AMT_BALANCE_max=('AMT_BALANCE_max',max),
                                                                                AMT_DRAWINGS_CURRENT=('AMT_DRAWINGS_CURRENT','mean'))
    
    return data_train,data_test

# on effectue du features engineering

def feat_eng():
    data_train,data_test = agglomération()
    
    # transformation de certaines colonnes en binaire
    data_test["FLAG_OWN_CAR"][data_test["FLAG_OWN_CAR"]=='N']=0
    data_test["FLAG_OWN_CAR"][data_test["FLAG_OWN_CAR"]=='Y']=1
    data_train["FLAG_OWN_CAR"][data_train["FLAG_OWN_CAR"]=='N']=0
    data_train["FLAG_OWN_CAR"][data_train["FLAG_OWN_CAR"]=='Y']=1

    data_test["FLAG_OWN_REALTY"][data_test["FLAG_OWN_REALTY"]=='N']=0
    data_test["FLAG_OWN_REALTY"][data_test["FLAG_OWN_REALTY"]=='Y']=1
    data_train["FLAG_OWN_REALTY"][data_train["FLAG_OWN_REALTY"]=='N']=0
    data_train["FLAG_OWN_REALTY"][data_train["FLAG_OWN_REALTY"]=='Y']=1
    
    # on récupère l'âge du client à partir de la variable DAYS_BIRTH
    data_test["AGE"] = abs(data_test["DAYS_BIRTH"])//365.25
    data_train["AGE"] = abs(data_train["DAYS_BIRTH"])//365.25
    data_test.drop(labels = 'DAYS_BIRTH', axis=1, inplace = True)
    data_train.drop(labels = 'DAYS_BIRTH', axis=1, inplace = True)
    
    # on garde le nombre de jours travaillés, mais en valeur absolue
    data_test["DAYS_EMPLOYED"] = abs(data_test["DAYS_EMPLOYED"])
    data_train["DAYS_EMPLOYED"] = abs(data_train["DAYS_EMPLOYED"])
    
    # on transforme la variable CREDIT_ACTIVE pour qu'elle donne le nombre de crédits qu'a contracté le client, 
    # et combien sont encore actifs
    data_test["AMT_CREDITS"] = data_test["CREDIT_ACTIVE"].str.len()
    data_train["AMT_CREDITS"] = data_train["CREDIT_ACTIVE"].str.len()
    liste = []
    for i in range(len(data_test)):
        liste += [data_test["CREDIT_ACTIVE"].iloc[i].count("Active")]
    data_test["AMT_CREDIT_ACTIVE"] = liste
    liste = []
    for i in range(len(data_train)):
        liste += [data_train["CREDIT_ACTIVE"].iloc[i].count("Active")]
    data_train["AMT_CREDIT_ACTIVE"] = liste
    data_test.drop(labels = 'CREDIT_ACTIVE', axis=1, inplace = True)
    data_train.drop(labels = 'CREDIT_ACTIVE', axis=1, inplace = True)
    
    # pour les crédits terminés, on remplace la valeur de la variable DAYS_CREDIT_ENDDATE par 0
    data_test["DAYS_CREDIT_ENDDATE"][data_test["DAYS_CREDIT_ENDDATE"] <= 0]=0
    data_train["DAYS_CREDIT_ENDDATE"][data_train["DAYS_CREDIT_ENDDATE"] <= 0]=0
    
    # on s'intéresse aux différents types de crédits qui ont été contracté en amont par le client
    data_test["NB_CREDIT_TYPE"] = data_test["CREDIT_TYPE"].str.len()
    data_train["NB_CREDIT_TYPE"] = data_train["CREDIT_TYPE"].str.len()
    
    # on convertie la variable NAME_CONTRACT_STATUS en binaire (1 si le crédit a été approuvé, 0 sinon)
    data_test["NAME_CONTRACT_STATUS"][data_test["NAME_CONTRACT_STATUS"] != 'Approved']=0
    data_train["NAME_CONTRACT_STATUS"][data_train["NAME_CONTRACT_STATUS"] != 'Approved']=0
    data_test["NAME_CONTRACT_STATUS"][data_test["NAME_CONTRACT_STATUS"] == 'Approved']=1
    data_train["NAME_CONTRACT_STATUS"][data_train["NAME_CONTRACT_STATUS"] == 'Approved']=1
    
    #on traite les variables liées aux versements (le nombre restant, la valeur payée, la valeur dûe)
    liste = []
    for i in range(len(data_test)):
        liste += [sum(x for x in data_test["CNT_INSTALMENT_FUTURE"].iloc[i] if is_nan(x)==False)]
    data_test["CNT_INSTALMENT_FUTURE"] = liste
    liste = []
    for i in range(len(data_train)):
        liste += [sum(x for x in data_train["CNT_INSTALMENT_FUTURE"].iloc[i] if is_nan(x)==False)]
    data_train["CNT_INSTALMENT_FUTURE"] = liste
    liste = []
    for i in range(len(data_test)):
        liste += [sum(x for x in data_test["AMT_INSTALMENT"].iloc[i] if is_nan(x)==False)]
    data_test["AMT_INSTALMENT"] = liste
    liste = []
    for i in range(len(data_train)):
        liste += [sum(x for x in data_train["AMT_INSTALMENT"].iloc[i] if is_nan(x)==False)]
    data_train["AMT_INSTALMENT"] = liste
    liste = []
    for i in range(len(data_test)):
        liste += [sum(x for x in data_test["AMT_PAYMENT"].iloc[i] if is_nan(x)==False)]
    data_test["AMT_PAYMENT"] = liste
    liste = []
    for i in range(len(data_train)):
        liste += [sum(x for x in data_train["AMT_PAYMENT"].iloc[i] if is_nan(x)==False)]
    data_train["AMT_PAYMENT"] = liste
    
    # on créé une variable donnant le reste à payer du versement du crédit précédent
    data_test["AMT_REST_TO_PAY"] = data_test["AMT_INSTALMENT"] - data_test["AMT_PAYMENT"]
    data_train["AMT_REST_TO_PAY"] = data_train["AMT_INSTALMENT"] - data_train["AMT_PAYMENT"]
    
    return data_test,data_train

# on traite les valeurs manquantes

def valeurs_manquantes():
    data_test,data_train = feat_eng()
    
    # on comble les Nan des variables les moins remplies, lorsque ça a du sens
    colonnes = ["CREDIT_DAY_OVERDUE","DAYS_CREDIT_ENDDATE","AMT_CREDIT_MAX_OVERDUE",]
    for col in colonnes:
        data_test[col][is_nan(data_test[col])]=0
        data_train[col][is_nan(data_train[col])]=0
        
    #on se sépare de certaines variables trop vides:
    liste = ['AMT_BALANCE_mean', 'AMT_BALANCE_min','AMT_BALANCE_max', 'AMT_DRAWINGS_CURRENT',"OWN_CAR_AGE"]
    data_test.drop(labels = liste, axis=1, inplace = True)
    data_train.drop(labels = liste, axis=1, inplace = True)
    
    # enfin, on supprime les entrées restantes qui contiennent des Nan:
    data_test.dropna(inplace = True)
    data_train.dropna(inplace = True)

    # correction des valeurs aberrantes:
    
    # on ne peut pas avoir un reste à payer négatif
    data_test["AMT_REST_TO_PAY"][data_test["AMT_REST_TO_PAY"]<0]=0
    data_train["AMT_REST_TO_PAY"][data_train["AMT_REST_TO_PAY"]<0]=0
    
    # on va considérer que l'age limite est 100 ans
    data_test = data_test[data_test["AGE"]<=100]
    data_train = data_train[data_train["AGE"]<=100]

    # on considère qu'on ne travaille pas plus de 18 250 jours (environ 50 ans)
    data_test = data_test[data_test['DAYS_EMPLOYED']<=18250]
    data_train = data_train[data_train['DAYS_EMPLOYED']<=18250]

    # on supprime les valeurs XNA dans les variables catégorielles:
    for col in ['CODE_GENDER','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 
             'ORGANIZATION_TYPE','NAME_CONTRACT_TYPE_PREV']:
        data_test = data_test[~(data_test[col]=='XNA')]
        data_train = data_train[~(data_train[col]=='XNA')]
    
    return data_test,data_train

# on prépare nos variables pour le modèle (passage au log, encoding, etc)

def pré_traitement():
    data_test,data_train = valeurs_manquantes()
    
    # modification des variables en log :
    colonnes = ['AMT_INCOME_TOTAL',
       'DAYS_EMPLOYED',  'CREDIT_DAY_OVERDUE',
       'DAYS_CREDIT_ENDDATE', 'AMT_CREDIT_MAX_OVERDUE', 'AMT_CREDIT_SUM_DEBT',
         'AMT_CREDIT_PREV',
        'CNT_INSTALMENT_FUTURE',
       'AMT_INSTALMENT', 'AMT_PAYMENT',  'AMT_CREDITS',
       'AMT_CREDIT_ACTIVE', 'AMT_REST_TO_PAY']
    for colonne in colonnes :
        data_test[colonne+"_log"] = np.log( 1 + data_test[colonne])
        data_train[colonne+"_log"] = np.log( 1 + data_train[colonne])
        
    # on convertie nos variables catégorielles en variables numériques
    colonnes = ['CODE_GENDER','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE', 
             'ORGANIZATION_TYPE','NAME_CONTRACT_TYPE_PREV']
    for col in colonnes :
        colonnes = pd.get_dummies(data_train[col])
        pca = decomposition.PCA(n_components= 1)
        data_train[col+"_num"] = pca.fit_transform(colonnes)
    
        colonnes = pd.get_dummies(data_test[col])
        pca = decomposition.PCA(n_components= 1)
        data_test[col+"_num"] = pca.fit_transform(colonnes)
        
    # on va garder uniquement certaines variables, afin de réduire le fléau de dimension
    variables_test = ['AMT_INCOME_TOTAL_log','REGION_RATING_CLIENT_W_CITY',
                 'ORGANIZATION_TYPE_num','NAME_CONTRACT_STATUS','DAYS_EMPLOYED_log',
                 'AMT_CREDIT_MAX_OVERDUE_log',
                 'AMT_CREDIT_ACTIVE_log','AMT_REST_TO_PAY_log','CODE_GENDER_num']
    
    return data_test,data_train,variables_test


# enfin, on prédit le défaut de paiement à partir de notre modèle pré-entrainé sur les données train (complétée via SMOTE)

def predict(arg):
    data_test,data_train,variables_test = pré_traitement()
    
    # on prépare nos données test pour la prédiction
    data_test.dropna(inplace=True)
    X = data_test[variables_test].values      
    colonnes = data_test[variables_test].columns
    std_scale = preprocessing.StandardScaler().fit(X)
    X = std_scale.transform(X)
    
    # on importe le modèle
    model = joblib.load('./P7_model.pkl') 
    
    # on effectue nos prédictions
    y_pred = model.predict(X)
    probas = model.predict_proba(X)
    data_test["Prédiction"] = y_pred
    data_test["Probabilité"] = probas[:,1]
    
    #on retourne le DF demandé en argument
    if arg == 'train':
        return data_train
    elif arg == 'test':
        return data_test
    else:
        return "Error : arg must be 'train' or 'test' !"

# Cette fonction finale renvoie les dataframes data_test et data_train
# data_test contient la prédiction finale (sous forme binaire) ainsi que la probabilité de défaut de paiement
# data_train reste utile pour faire des comparaisons 



    
    
