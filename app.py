#  Airbnb Price Prediction Using Machine Learning and Analysing Sentiment from guest review

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


st.set_page_config(
    page_title="Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide"
)

tab1, tab2 = st.tabs(
    [
        "🏠 Price Prediction",
        "😊 Sentiment Analysis"
    ]
)

with tab1:

    st.header("🏠 Airbnb Price Prediction")

    # Load model
    model = joblib.load("airbnb_price_model.pkl")

    neighbourhood_map = joblib.load("neighbourhood_map.pkl")

    neighbourhood_coords = joblib.load("neighbourhood_coords.pkl")

    st.write("Estimate Airbnb listing prices in New York City.")
    
    col1, col2 = st.columns(2)

    with col1:

        room_type = st.selectbox(
            "Room Type",
            [
                "Entire home/apt",
                "Private room",
                "Shared room"
            ]
        )

        neighbourhood = st.selectbox(
            "Neighbourhood",
            sorted(neighbourhood_map.index)
        )

        minimum_nights = st.number_input(
            "Minimum Nights",
            min_value=1
        )

        number_of_reviews = st.number_input(
            "Number of Reviews",
            min_value=0
        )

    latitude = neighbourhood_coords.loc[
        neighbourhood,
        "latitude"
    ]

    longitude = neighbourhood_coords.loc[
        neighbourhood,
        "longitude"
    ]

    with col2:

        reviews_per_month = st.number_input(
            "Reviews Per Month",
            min_value=0.0
        )

        calculated_host_listings_count = st.number_input(
            "Host Listings Count",
            min_value=1
        )

        availability_365 = st.number_input(
            "Availability 365",
            min_value=0,
            max_value=365
        )

    room_type_private = (
        1 if room_type == "Private room" else 0
    )

    room_type_shared = (
        1 if room_type == "Shared room" else 0
    )

    neighbourhood_encoded = neighbourhood_map.get(
        neighbourhood,
        neighbourhood_map.median()
    )

    st.markdown("""
    <style>
    div.stButton > button {
        background-color: #28a745;
        color: white;
        font-size: 20px;
        font-weight: bold;
        border-radius: 10px;
    }

    div.stButton > button:hover {
        background-color: #218838;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Predict Price", use_container_width=True):

        input_df = pd.DataFrame({
            "latitude": [latitude],
            "longitude": [longitude],
            "minimum_nights": [minimum_nights],
            "number_of_reviews": [number_of_reviews],
            "reviews_per_month": [reviews_per_month],
            "calculated_host_listings_count": [calculated_host_listings_count],
            "availability_365": [availability_365],
            "room_type_Private room": [room_type_private],
            "room_type_Shared room": [room_type_shared],
            "neighbourhood_encoded": [neighbourhood_encoded]
        })

        prediction = model.predict(input_df)[0]

        price = round(np.expm1(prediction), 2)

        st.metric(
            label="Estimated Airbnb Price",
            value=f"${price:,.0f}"
        )

        # st.info(
        #     "Latitude and longitude are automatically assigned "
        #     "using the median coordinates of the selected neighbourhood."
        # )




with tab2:

    st.header("😊 Guest Review Sentiment Analysis")

    st.write("Enter an Airbnb guest review to analyze its sentiment.")


    review = st.text_area(
        "Guest Review",
        height=150
    )

    if st.button(
        " Analyze Sentiment",
        use_container_width=True
    ):

        score = analyzer.polarity_scores(review)["compound"]

        if score >= 0.05:
            sentiment = "Positive 😊"

        elif score <= -0.05:
            sentiment = "Negative 😞"

        else:
            sentiment = "Neutral 😐"

        st.metric(
            label="Sentiment",
            value=sentiment
        )

        st.write(
            f"Sentiment Score: {score:.2f}"
        )

       