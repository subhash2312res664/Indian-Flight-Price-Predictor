import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime

# Load the trained model
try:
    with open('flight_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please make sure 'flight_price_model.pkl' is in the same directory.")
    st.stop()

# Define the feature mapping based on the training script
# This is crucial to ensure the input to the model is in the correct format.
# CORRECTED: Added 'Air Asia' to the list to match the training data, for a total of 12 airlines.
airlines = [
    'Air Asia', 'Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business',
    'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet', 'Trujet',
    'Vistara', 'Vistara Premium economy'
]
sources = ['Banglore', 'Chennai', 'Delhi', 'Kolkata', 'Mumbai']
destinations = ['Banglore', 'Cochin', 'Delhi', 'Hyderabad', 'Kolkata', 'New Delhi']

# --- Streamlit App UI ---

st.set_page_config(page_title="Flight Price Predictor", page_icon="✈️", layout="wide")

st.title("✈️ Indian Flight Price Predictor")
st.markdown("This app predicts the price of a flight ticket based on your travel details. It uses a machine learning model trained on a real-world dataset.")

st.sidebar.header("Enter Your Flight Details")

# --- User Inputs ---

# Journey Date
journey_date = st.sidebar.date_input("Date of Journey", datetime.date.today())
journey_day = journey_date.day
journey_month = journey_date.month

# Departure and Arrival Times
dep_time = st.sidebar.time_input("Departure Time", datetime.time(9, 45))
dep_hour = dep_time.hour
dep_min = dep_time.minute

arrival_time = st.sidebar.time_input("Arrival Time", datetime.time(19, 10))
arrival_hour = arrival_time.hour
arrival_min = arrival_time.minute

# Duration Calculation (for display and model input)
duration_hours = arrival_hour - dep_hour
duration_mins = arrival_min - dep_min
if duration_mins < 0:
    duration_hours -= 1
    duration_mins += 60
if duration_hours < 0:
    duration_hours += 24

duration_in_mins = duration_hours * 60 + duration_mins
st.sidebar.info(f"Calculated Duration: {duration_hours}h {duration_mins}m")


# Categorical Inputs
airline = st.sidebar.selectbox("Airline", sorted(airlines))
source = st.sidebar.selectbox("Source", sorted(sources))
destination = st.sidebar.selectbox("Destination", sorted(destinations))
total_stops = st.sidebar.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])

# Mapping stops to numerical values
stops_mapping = {'non-stop': 0, '1 stop': 1, '2 stops': 2, '3 stops': 3, '4 stops': 4}
total_stops_numeric = stops_mapping[total_stops]


# --- Prediction Logic ---
if st.sidebar.button("Predict Price", type="primary"):

    # 1. Create the input array in the correct order
    # The order must match the columns in the training data (X_train)

    # Start with numerical features
    input_data = [
        total_stops_numeric,
        journey_day,
        journey_month,
        dep_hour,
        dep_min,
        arrival_hour,
        arrival_min,
        duration_in_mins
    ]

    # 2. Handle One-Hot Encoding for categorical features
    # Airline
    airline_vector = [0] * len(airlines)
    if airline in airlines:
        airline_vector[airlines.index(airline)] = 1
    input_data.extend(airline_vector)

    # Source
    source_vector = [0] * len(sources)
    if source in sources:
        source_vector[sources.index(source)] = 1
    input_data.extend(source_vector)

    # Destination
    destination_vector = [0] * len(destinations)
    if destination in destinations:
        destination_vector[destinations.index(destination)] = 1
    input_data.extend(destination_vector)

    # 3. Make the prediction
    prediction = model.predict([input_data])
    predicted_price = round(prediction[0])

    # 4. Display the result
    st.success(f"## Predicted Flight Price: ₹ {predicted_price:,}")

    st.markdown("---")
    st.subheader("Your Travel Details:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Airline:** {airline}")
        st.write(f"**Source:** {source}")
        st.write(f"**Destination:** {destination}")
    with col2:
        st.write(f"**Journey Date:** {journey_date.strftime('%d %B, %Y')}")
        st.write(f"**Departure Time:** {dep_time.strftime('%H:%M')}")
        st.write(f"**Arrival Time:** {arrival_time.strftime('%H:%M')}")
    with col3:
        st.write(f"**Duration:** {duration_hours}h {duration_mins}m")
        st.write(f"**Total Stops:** {total_stops.title()}")


st.markdown("""
---
**Note:** This prediction is for demonstration purposes. The model was trained on data from March-June 2019 for domestic Indian flights.
""")

