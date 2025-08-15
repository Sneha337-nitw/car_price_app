import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings

# This line tells Python to ignore all warning messages that would normally appear in the output.
warnings.filterwarnings("ignore")

# Load the trained model
# The file needs to be in the same directory as this app.py file
try:
    with open('random_forest_regression_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(
        "Error: The model file 'random_forest_regression_model.pkl' was not found. Please ensure it is in the same directory as this script.")
    st.stop()

# Load the original dataset for the suggestion feature
try:
    df = pd.read_csv('lets try 1 car.csv')
    # Add a 'brand' column by extracting the first word from the 'name' column
    df['brand'] = df['name'].apply(lambda x: x.split(' ')[0])
except FileNotFoundError:
    st.error(
        "Error: The dataset file 'lets try 1 car.csv' was not found. Please ensure it is in the same directory as this script.")
    st.stop()

# Set up the Streamlit app layout
st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Main title of the app
st.title("Car Price & Suggestion App")
st.markdown("---")

# Use a radio button in the sidebar to switch between features
app_mode = st.sidebar.radio("Choose the app mode", ["Car Price Predictor", "Car Suggestion"])

if app_mode == "Car Price Predictor":
    st.header("Predict the Selling Price of a Car")
    st.write("Please enter the details of the car to get a price prediction.")

    # Create two columns for a better layout
    col1, col2 = st.columns(2)

    # Column 1 for input fields
    with col1:
        # Get user input for car features.
        # The default values and options are based on the data in the provided notebook.

        # Current year for calculating the age of the car
        current_year = 2020

        # Car Year - The year the car was purchased
        year = st.selectbox(
            'Car Year (When was the car purchased?)',
            options=list(range(1983, current_year + 1)),
            index=len(list(range(1983, current_year + 1))) - 1
        )

        # Kilometers Driven
        km_driven = st.slider('Kilometers Driven', min_value=1, max_value=2500000, value=70000, step=1000)

        # Fuel Type - This maps directly to the one-hot encoded columns
        fuel_options = ['Diesel', 'Petrol', 'CNG', 'LPG']
        fuel = st.selectbox('Fuel Type', options=fuel_options)

        # Seller Type
        seller_options = ['Individual', 'Dealer', 'Trustmark Dealer']
        seller = st.selectbox('Seller Type', options=seller_options)

        # Transmission Type
        transmission_options = ['Manual', 'Automatic']
        transmission = st.selectbox('Transmission', options=transmission_options)

        # Car Ownership
        owner_options = ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car']
        owner = st.selectbox('Owner', options=owner_options)

    with col2:
        # Mileage (in kmpl or km/kg)
        mileage = st.number_input('Mileage (kmpl or km/kg)', min_value=0.0, max_value=50.0, value=18.0, step=0.1)

        # Engine displacement (in CC)
        engine = st.number_input('Engine (in CC)', min_value=1, max_value=5000, value=1500, step=1)

        # Maximum Power (in bhp)
        max_power = st.number_input('Max Power (in bhp)', min_value=1.0, max_value=500.0, value=100.0, step=1.0)

        # Number of seats in the car
        seats = st.number_input('Seats', min_value=1.0, max_value=10.0, value=5.0, step=1.0)

    st.markdown("---")
    if st.button('Predict Price', use_container_width=True):
        # Calculate the age of the car (no_year)
        no_year = current_year - year

        # Create a dictionary to hold the input values
        input_data = {
            'km_driven': km_driven,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seats': seats,
            'no_year': no_year,
            'fuel_CNG': 0, 'fuel_Diesel': 0, 'fuel_LPG': 0, 'fuel_Petrol': 0,
            'seller_type_Dealer': 0, 'seller_type_Individual': 0, 'seller_type_Trustmark Dealer': 0,
            'transmission_Automatic': 0, 'transmission_Manual': 0,
            'owner_First Owner': 0, 'owner_Fourth & Above Owner': 0, 'owner_Second Owner': 0, 'owner_Test Drive Car': 0,
            'owner_Third Owner': 0
        }

        # Apply one-hot encoding logic based on user selections
        input_data[f'fuel_{fuel}'] = 1
        input_data[f'seller_type_{seller}'] = 1
        input_data[f'transmission_{transmission}'] = 1
        input_data[f'owner_{owner}'] = 1

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([input_data])

        # The order of columns in the training data is critical for the model.
        # The code below ensures the user input matches the model's expected feature order.
        required_columns = ['km_driven', 'mileage', 'engine', 'max_power', 'seats', 'no_year',
                            'fuel_CNG', 'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol',
                            'seller_type_Dealer', 'seller_type_Individual', 'seller_type_Trustmark Dealer',
                            'transmission_Automatic', 'transmission_Manual',
                            'owner_First Owner', 'owner_Fourth & Above Owner', 'owner_Second Owner',
                            'owner_Test Drive Car', 'owner_Third Owner']

        input_df = input_df[required_columns]

        # Make the prediction
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"The predicted selling price of the car is: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif app_mode == "Car Suggestion":
    st.header("Find Cars based on your criteria")
    st.write("You can search by price range, car brand, or both.")

    # Get user input for filters
    col1_search, col2_search = st.columns(2)
    with col1_search:
        # Car Brand - The first word of the car name is treated as the brand
        brand_options = ['All'] + sorted(df['brand'].unique().tolist())
        selected_brand = st.selectbox('Car Brand', options=brand_options)
    with col2_search:
        use_price_range = st.checkbox("Filter by Price Range")

    min_selling_price = 0
    max_selling_price = df['selling_price'].max()

    if use_price_range:
        min_selling_price = st.number_input('Minimum Price (₹)', min_value=0)
        max_selling_price = st.number_input('Maximum Price (₹)', min_value=min_selling_price + 1, value=500000)

    st.markdown("---")
    if st.button('Search for Cars', use_container_width=True):
        suggested_cars = df.copy()

        # Filter by brand if a specific brand is selected
        if selected_brand != 'All':
            suggested_cars = suggested_cars[suggested_cars['brand'] == selected_brand]

        # Filter by price range if the checkbox is selected
        if use_price_range:
            suggested_cars = suggested_cars[
                (suggested_cars['selling_price'] >= min_selling_price) &
                (suggested_cars['selling_price'] <= max_selling_price)
                ]

        if not suggested_cars.empty:
            # Define the columns to display
            columns_to_display = ['name', 'selling_price', 'fuel', 'km_driven', 'seats', 'engine']
            # Display the results with only the selected columns
            st.subheader(f"Found {len(suggested_cars)} cars matching your criteria:")
            st.dataframe(suggested_cars[columns_to_display])
        else:
            st.info("No cars found matching the specified criteria. Please try a different search.")
