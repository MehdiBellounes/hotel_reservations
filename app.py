# -*- coding: utf-8 -*-
# importe les bibliotheques de donnÃ©es
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
# import le module permettant de rajouter une image
from PIL import Image
# import le module permettant la manipulation des dates
from datetime import datetime, date, timedelta
# import le module permettant l'integration d'un fichier pickle
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px # Using express
import plotly.graph_objects as go # Using go.Figure() ...
from plotly.subplots import make_subplots # Custom subplots and double axes
tabs = ["Presentation projet", "Visualisation projet", "Prediction"]

url = pd.read_csv("Hotel Reservations.csv")
df = pd.DataFrame(url)
df['booking_status'] = df['booking_status'].replace({"Not_Canceled": 0, "Canceled": 1}) 
#ajout d'une colonne mois et transformation des 29/02
mask = (df["arrival_month"] == 2) & (df["arrival_date"]==29)
df.loc[mask,"arrival_date"]=28
df["date"] = df['arrival_year'].astype(str) + "-"+ df["arrival_month"].astype(str) +"-" +df["arrival_date"].astype(str)
df["date"] = pd.to_datetime(df["date"])

#Visualisation
#Evolution des taux d'annulation dans le temps
fig1 = px.line(df.set_index('date').resample("W").mean().reset_index(), x="date", y="booking_status", title="Evolution des taux d'annulation dans le temps",template = 'plotly_dark',labels={'booking_status':"Taux d'annulation"})
#Taux d'annulation en fonction de la durée entre la date réservation et la date d'arrivée
df["lead_time_bins"] = pd.cut(df.lead_time, np.arange(0, 500, 50))#créer des ranges"bins" pour le lead time de 50 en 50
df["lead_time_bins_text"] = df["lead_time_bins"].map(lambda x: f"{x.left}-{x.right}")#transformer ces ranges en string (car plotly n'accepte pas les bins dans les graphiques)
fig2 = px.bar(df.groupby(by='lead_time_bins_text').mean().reset_index(), x='lead_time_bins_text', y='booking_status',labels={"lead_time_bins_text":"Durée entre la date réservation et la date d'arrivée", "booking_status": "Taux (%) d'annulation"},title="Taux d'annulation en fonction de la durée entre la date réservation et la date d'arrivée", template = 'plotly_dark')
#Taux d'annulation en fonction du prix de la chambre
df["avg_price_per_room_bins"] = pd.cut(df.avg_price_per_room, np.arange(0, 600, 10))#créer des ranges"bins" pour le prix des chambres de 50 en 50
df["avg_price_per_room_bins_text"] = df["avg_price_per_room_bins"].map(lambda x: f"{x.left}-{x.right}")#transformer ces ranges en string (car plotly n'accepte pas les bins dans les graphiques)
fig3 = px.bar(df.groupby(by='avg_price_per_room_bins_text').mean().reset_index(), x='avg_price_per_room_bins_text', y='booking_status',labels={"avg_price_per_room_bins_text":"Prix de la chambre", "booking_status": "Taux (%) d'annulation"},title="Taux d'annulation en fonction du prix de la chambre", template = 'plotly_dark')

select_tab = st.sidebar.radio("Choisir l'onglet", tabs)
if select_tab == "Prediction":
    # apelle le module titre de la bibliotheque streamlit
    image = Image.open('image_drole.jpeg')
    st.image(image,width = 500)
    st.title("Prediction de l'annulation de la chambre")

    # import image

    

    # choisi un titre pour ton app

    st.header("Entrez vos criteres pour savoir si la chambre va etre annulée")
    # permet d'inserer une image que tu as importÃ©
    
    # crÃ©e une selectbox qui te permet de faire plusieurs choix

    no_of_adults = st.select_slider(
        "Nombre d'adultes", options=(0, 1, 2, 3, 4, 5, 6, 7), value=2)
    no_of_children = st.select_slider(
        "Nombre d'enfants", options=(0, 1, 2, 3, 4, 5, 6, 7), value=0)

    #date_reservation = st.date_input("quel est la date de reservation?")

    # module liste dÃ©roulante pour choix du type de repas et stockage dans

    type_of_meal_plan = st.selectbox('Quel type de repas', options=(
        'Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'))
    required_car_parking_space = st.selectbox('Parking?', options=('0', '1'))
    room_type_reserved = st.selectbox('Quel type de chambre?', options=(
        'Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'))

    # # module date outpout => datetype format yea-mon-date

    date_arrivee = st.date_input("quel est la date d'arrivage?")
    date_depart = st.date_input("quel est la date de depart?")

    # calcul du nombre de jour entre la date d'arrivee et la date de reservation - major feature -

    lead_time = (date_arrivee - datetime.today().date()).days

    # stockage dans la variable  de l'input user

    arrival_year = date_arrivee.year
    arrival_month = date_arrivee.month
    arrival_date = date_arrivee.day

    # definition des variables generique pour integration dataframe

    market_segment_type = "Online"
    repeated_guest = 0
    no_of_previous_cancellations = 0
    no_of_previous_bookings_not_canceled = 0
    room_type_price_mapping = {'Room_Type 1': 96, 'Room_Type 2': 88, 'Room_Type 3': 74,
                               'Room_Type 4': 125, 'Room_Type 5': 124, 'Room_Type 6': 182, 'Room_Type 7': 155}
    avg_price_per_room = room_type_price_mapping[room_type_reserved]
    options = ["ascenseur", "lit bebe", "tv", "playstation", "sto"]

    # module de choix des critÃ¨res pour saisie utilisateur

    special_requests = st.multiselect("selectionnez les options", options)

    # stockage dans la variable de l'input nombre de choix selectionnÃ©s par user

    no_of_special_requests = len(special_requests)

    #  calcule du nombre de nuit par week end et semaine pour integration dans dataframe

    no_of_week_nights = 0
    no_of_weekend_nights = 0
    delta = timedelta(days=1)

    while date_arrivee < date_depart:
        if date_arrivee.weekday() < 5:
            no_of_week_nights += 1
        else:
            no_of_weekend_nights += 1
        date_arrivee += delta

    print(arrival_year)

    # crÃ©ation du dataframe recevant les variables des inputs user par l'appli

    booking = pd.DataFrame({
        "no_of_adults": [no_of_adults],
        "no_of_children": [no_of_children],
        "type_of_meal_plan": [type_of_meal_plan],
        "required_car_parking_space": [required_car_parking_space],
        "room_type_reserved": [room_type_reserved],
        "lead_time": [lead_time],
        "arrival_year": [arrival_year],
        "arrival_month": [arrival_month],
        "arrival_date": [arrival_date],
        "market_segment_type": [market_segment_type],
        "repeated_guest": [repeated_guest],
        "no_of_previous_cancellations": [no_of_previous_cancellations],
        "no_of_previous_bookings_not_canceled": [no_of_previous_bookings_not_canceled],
        "avg_price_per_room": [avg_price_per_room],
        "no_of_special_requests": [no_of_special_requests],
        "no_of_week_nights": [no_of_week_nights],
        "no_of_weekend_nights": [no_of_weekend_nights]
    })

    # identifications des erreurs de saisie du 29/02 remplacement par 28
    # identification du jour
    # creation colonne

    def replace_29(df_input):
        df = df_input.copy()
        mask = (df["arrival_month"] == 2) & (df["arrival_date"] == 29)
        df.loc[mask, "arrival_date"] = 28
        df["date"] = df['arrival_year'].astype(
            str) + "-" + df["arrival_month"].astype(str) + "-" + df["arrival_date"].astype(str)
        df["date"] = pd.to_datetime(df["date"])
        df["jour"] = df["date"].dt.dayofweek
        return df[["jour"]]

    # import fichier pickle dans la base
    with open('hotel_reservation_rf_model.pkl', 'rb') as f:
        rfmodel = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    #
    if st.button("predictions"):
        st.write(label_encoder.inverse_transform(rfmodel.predict(booking)))
elif select_tab == "Presentation projet":

    st.header("Fichier hotel reservation")
    image_hotel = Image.open('image_hotel.jpg')
    st.image(image_hotel)
    st.markdown("""Suite a l augmentation des annulations pour un hotel , il a decide de nous contacter pour travailler sur ses donnees de reservation avec pour objectif de predire si la reservation va etre annulee par le client""")
    st.plotly_chart(fig1)
elif select_tab == "Visualisation projet":

    
    st.header("Fichier hotel reservation")
    
    st.dataframe(df.head(5))
    st.markdown(
"""Nous avons recupere des donnees provenant d un hotel.
Ces donnees proviennent de reservations faites par des clients.
\
Nous avons differents **types de donnees**: 
**information du clients** (historique d annulation)
**information chambre**
**indicateurs lies e la reservation** (duree sejour , nombre enfant ,adultes)
""")
    
    st.plotly_chart(fig2)
    st.markdown("Suite au raffinage de nos donnees nous avons observes que plus une chambre est reservee a l avance plus il y a de chance qu elle soit annulee.")
    st.markdown("**c’est le critere majeur d annulation sur les reservations.**")
    st.plotly_chart(fig3)
    st.markdown("Nous ne constatons **pas de correlation entre le prix de la chambre et le taux d annulation.**")
    
    st.header("Choix du model")
    st.markdown("Le modele **random forest** est un algorithme qui utilise plusieurs arbres de decision pour **predire des valeurs**. Il est connu pour sa robustesse et sa capacite à traiter des donnees complexes en moyennant les predictions de chaque arbre pour donner une prediction finale.")
    image_cvs = Image.open('matrice_cross_val.jpg')
    st.image(image_cvs)
    image_recall = Image.open('cross_validate_recall.jpg')
    st.image(image_recall)
    st.markdown("**La matrice de confusion** est un outil pour evaluer **la qualite d'un modele de classification** en comparant ses predictions avec les vraies etiquettes d'un ensemble de donnees. La matrice de confusion **permet de calculer des mesures de performance** importantes pour evaluer la qualite du modele, telles que la precision et le rappel.")
    
