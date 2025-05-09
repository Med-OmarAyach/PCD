from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.contrib import messages
from .forms import CustomUserCreationForm
from django.contrib.auth.decorators import login_required
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from django.http import JsonResponse
from rest_framework.decorators import api_view
import os
from datetime import datetime
import time
import threading

# Global Variables
CSV_FILENAME = 'data/data_X_anom.csv'
DATA_ITERATOR = None
CURRENT_DATA_INDEX = 0
DATA_LOCK = threading.Lock()
USING_SAMPLE_DATA = False  # Flag to indicate if we're using sample data

# Load pre-trained models at application startup
autoencoders = {
    1: tf.keras.models.load_model('models/autoencoder_modelG1Room1.keras'),
    2: tf.keras.models.load_model('models/autoencoder_modelG1Room2.keras'),
    3: tf.keras.models.load_model('models/autoencoder_modelG1Room4.keras'),
    4: tf.keras.models.load_model('models/autoencoder_modelG1Room5.keras'),
}

# Thresholds for anomaly detection for each room
thresholds = {
    1: 3,  # Room 1
    2: 3,  # Room 2
    3: 3,  # Room 3
    4: 3,  # Room 4
}

clf_stage1 = joblib.load('models/xgboost_stage1_model.pkl')
clf_stage2 = joblib.load('models/svm_stage2_model.pkl')

# Room data cache (only stores the current state)
CURRENT_ROOM_DATA = {
    1: None,  # Room 1 data
    2: None,  # Room 2 data
    3: None,  # Room 3 data
    4: None,  # Room 4 data
}

# Room history cache (stores recent history (max 30 entries) )
ROOM_HISTORY = {
    1: [],  # Room 1 history
    2: [],  # Room 2 history
    3: [],  # Room 3 history
    4: [],  # Room 4 history
}

# Maximum history entries per room
MAX_HISTORY_ENTRIES = 30

def sample_data_generator():
    """Generate sample data one row at a time"""
    while True:
        # Generate a single row of sample data
        row = {}
        row['date_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate data for each room
        for room_num in range(1, 5):
            # Base temperature with some randomness
            base_temp = 22.0 + np.random.normal(0, 2)

            # Add temperature columns for each sensor in the room
            for sensor_num in range(1, 4):
                col_name = f"T_data_{room_num}_{sensor_num}"
                # Add some variation between sensors
                row[col_name] = round(base_temp + np.random.normal(0, 0.5), 1)

            # Random anomaly assignment - 80% normal, 20% anomalies
            if np.random.random() < 0.8:
                row[f"Anomaly_Label_R{room_num}"] = "Normal"
            else:
                anomaly_types = ["Network", "Security", "Sensor"]
                row[f"Anomaly_Label_R{room_num}"] = np.random.choice(anomaly_types)

        # Yield as a DataFrame with a single row
        yield pd.DataFrame([row])

        # Simulate delay between data points in the generator (not needed in production)
        time.sleep(0.1)

def initialize_data():
    #Initialize the data iterator - should be called on application startup
    global DATA_ITERATOR, CURRENT_DATA_INDEX, USING_SAMPLE_DATA

    try:
        # Check if file exists
        if os.path.exists(CSV_FILENAME):
            # Open the file as an iterator to avoid loading it all in memory
            df = pd.read_csv(CSV_FILENAME)
            print(f"CSV file loaded successfully: {CSV_FILENAME} with {len(df)} rows")
            DATA_ITERATOR = df
            USING_SAMPLE_DATA = False
        else:
            # If file doesn't exist, use sample data generator
            print(f"CSV file not found at {CSV_FILENAME}, using sample data generator")
            # Create a sample DataFrame with 1000 rows to simulate CSV data
            sample_rows = []
            gen = sample_data_generator()
            for _ in range(1000):
                sample_rows.append(next(gen).iloc[0].to_dict())
            DATA_ITERATOR = pd.DataFrame(sample_rows)
            USING_SAMPLE_DATA = True

        # Initialize with the first data point
        update_current_data()

        # Start the background thread for automatic updates
        background_thread = threading.Thread(target=periodic_data_update, daemon=True)
        background_thread.start()

    except Exception as e:
        print(f"Error initializing data: {str(e)}")
        # Create a sample DataFrame as fallback
        sample_rows = []
        gen = sample_data_generator()
        for _ in range(1000):
            sample_rows.append(next(gen).iloc[0].to_dict())
        DATA_ITERATOR = pd.DataFrame(sample_rows)
        USING_SAMPLE_DATA = True
        update_current_data()

def periodic_data_update():
    #Background thread function to update data periodically
    while True:
        # Wait for 60 seconds
        time.sleep(60)

        try:
            # Update the current data
            update_current_data()
            print(f"Data updated automatically at {datetime.now()}")
        except Exception as e:
            print(f"Error in periodic update: {str(e)}")

def update_current_data():
    #Update the current data from the iterator
    global CURRENT_DATA_INDEX, DATA_ITERATOR

    with DATA_LOCK:
        try:
            # Get the next chunk from the iterator
            if CURRENT_DATA_INDEX >= len(DATA_ITERATOR):
                CURRENT_DATA_INDEX = 0  # Reset to beginning if we reached the end

            df_chunk = DATA_ITERATOR.iloc[[CURRENT_DATA_INDEX]]
            CURRENT_DATA_INDEX += 1

            # Process the data through anomaly detection
            processed_data = process_data_row(df_chunk)

            # Update the current data and history
            update_room_data_and_history(processed_data)
        except Exception as e:
            print(f"Error updating data: {str(e)}")

def process_data_row(df_row):
    """Process a single row of data through the anomaly detection models"""
    result_row = df_row.copy()

    for room_num in range(1, 5):
        try:
            # Get sensor column names for the room
            t_cols = [f"T_data_{room_num}_1", f"T_data_{room_num}_2", f"T_data_{room_num}_3"]

            # Ensure all columns exist
            for col in t_cols:
                if col not in result_row.columns:
                    result_row[col] = 225  # Default temperature

            # Get the data for model input
            datetime_val = result_row['date_time'].astype(np.float32).values.reshape(-1, 1)
            temp_data = result_row[t_cols].astype(np.float32).values

            # Concatenate temperatures with datetime (final shape should be (1, 4))
            room_data = np.concatenate([temp_data, datetime_val], axis=1)
            # Apply anomaly detection
            autoencoder = autoencoders.get(room_num)
            threshold = thresholds.get(room_num, 3)

            if autoencoder:
                # Use the autoencoder for reconstruction
                reconstructed = autoencoder.predict(room_data, verbose=0)
                error = np.mean(np.square(room_data - reconstructed))
                # Determine if it's an anomaly
                if error < threshold:
                    anomaly_label = "Normal"
                else:
                    # Calculate error features for classifier stage 1
                    error_features = np.array([
                        np.mean((room_data - reconstructed)**2),
                        np.std((room_data - reconstructed)**2),
                        np.max((room_data - reconstructed)**2),
                        np.min((room_data - reconstructed)**2)
                    ]).reshape(1, -1)

                    # Apply stage 1 classification (Network vs Other)
                    stage1_pred = clf_stage1.predict(error_features)[0]

                    if stage1_pred == 0:
                        anomaly_label = "Network"
                    else:
                        # Apply stage 2 classification (Security vs Sensor)
                        stage2_pred = clf_stage2.predict(room_data.reshape(1, -1))[0]

                        if stage2_pred == 1:
                            anomaly_label = "Security"
                        elif stage2_pred == 2:
                            anomaly_label = "Sensor"
                        else:
                            anomaly_label = "Unknown"
                print(f"Processing Room {room_num} - Error: {error} - Label: {anomaly_label}")
            else:
                # If no model is available, default to Normal
                anomaly_label = "Normal"

            # Store the anomaly label
            result_row[f"Anomaly_Label_R{room_num}"] = anomaly_label

        except Exception as e:
            print(f"Error processing room {room_num}: {str(e)}")
            # Default to Normal if there's any error
            result_row[f"Anomaly_Label_R{room_num}"] = "Normal"


    return result_row

def update_room_data_and_history(df_row):
    """Update the current room data and append to history"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    for room_num in range(1, 5):
        # Get temperature values
        t_cols = [f"T_data_{room_num}_1", f"T_data_{room_num}_2", f"T_data_{room_num}_3"]
        temps = [float(df_row[col].iloc[0]) if col in df_row.columns else 22.0 for col in t_cols]

        # Get anomaly label
        anomaly_key = f"Anomaly_Label_R{room_num}"
        anomaly_type = df_row[anomaly_key].iloc[0] if anomaly_key in df_row.columns else "Normal"

        # Create room data dict
        room_data = {
            "timestamp": timestamp,
            "temperature1": round(temps[0], 1),
            "temperature2": round(temps[1], 1),
            "temperature3": round(temps[2], 1),
            anomaly_key: anomaly_type
        }

        # Update current data
        CURRENT_ROOM_DATA[room_num] = room_data

        # Append to history and maintain max length
        ROOM_HISTORY[room_num].append(room_data)
        if len(ROOM_HISTORY[room_num]) > MAX_HISTORY_ENTRIES:
            ROOM_HISTORY[room_num].pop(0)

@api_view(['GET'])
def get_room_data(request):
    #API endpoint to get current room data
    # Force update if requested
    force_update = request.GET.get('force_update', 'false').lower() == 'true'
    if force_update:
        try:
            update_current_data()
        except Exception as e:
            print(f"Error forcing update: {str(e)}")

    # Prepare data to send to the frontend
    room_data = []

    with DATA_LOCK:
        for room_num in range(1, 5):
            room_info = CURRENT_ROOM_DATA.get(room_num)

            if room_info:
                # Use cached data
                room_data.append({
                    "room": f"ROOM {room_num}",
                    "temperature1": room_info["temperature1"],
                    "temperature2": room_info["temperature2"],
                    "temperature3": room_info["temperature3"],
                    f"Anomaly_Label_R{room_num}": room_info[f"Anomaly_Label_R{room_num}"],
                    "timestamp": room_info["timestamp"]
                })
            else:
                # If no data is available, provide defaults
                room_data.append({
                    "room": f"ROOM {room_num}",
                    "temperature1": 225,
                    "temperature2": 227,
                    "temperature3": 223,
                    f"Anomaly_Label_R{room_num}": "Normal",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })

    return JsonResponse({
        "room_data": room_data,
        "current_idx": CURRENT_DATA_INDEX,
        "next_update": int((time.time() % 60)),  # Seconds until next update
        "data_source": "CSV" if not USING_SAMPLE_DATA else "Sample Data"  # Indicates the data source
    })

@api_view(['GET'])
def get_room_history(request):
    #API endpoint to get room history data
    room_id = request.GET.get('room', '1')  # Default to room 1 if not provided

    # Validate room_id
    try:
        room_num = int(room_id)
        if room_num < 1 or room_num > 4:
            room_num = 1  # Default to room 1 if invalid
    except ValueError:
        room_num = 1

    # Get the history data for the requested room
    with DATA_LOCK:
        history = ROOM_HISTORY.get(room_num, [])

    return JsonResponse({
        "history": history,
        "data_source": "CSV" if not USING_SAMPLE_DATA else "Sample Data"  # Indicates the data source
    })

@api_view(['GET'])
def test_api_view(request):
    #API endpoint for testing if the API connectivity works
    return JsonResponse({
        "status": "success",
        "message": "API is working correctly",
        "current_idx": CURRENT_DATA_INDEX,
        "data_source": "CSV" if not USING_SAMPLE_DATA else "Sample Data"
    })

def signin_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        user = authenticate(request, username=email, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, 'Successfully logged in.')
            return redirect('main_page')
        else:
            messages.error(request, 'Invalid email or password.')

    return render(request, 'sign_in.html')

def signup_view(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            try:
                form.save()  # Automatically handles password hashing
                messages.success(request, 'Account created successfully!')
                return redirect('sign_in')
            except Exception as e:
                messages.error(request, f'Error creating account: {str(e)}')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f"{field}: {error}")
    else:
        form = CustomUserCreationForm()

    return render(request, 'sign_up.html', {'form': form})

@login_required
def main_page_view(request):
    return render(request, 'main_page.html')

# Initialize data when the module is loaded
initialize_data()


