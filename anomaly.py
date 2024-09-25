# Importing necessary libraries
import os
import random
import time
from datetime import datetime
from joblib import load
import logging
import matplotlib.pyplot as plt
import numpy as np
from settings import DELAY, OUTLIERS_GENERATION_PROBABILITY, VISUALIZATION, MAX_DATA_POINTS

# Configuring logging to write to a file named 'anomaly.log'
logging.basicConfig(filename='anomaly.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# List to store incoming real time data
data_ls = []

def anomaly_dect():
    _id = 0

    # Visualization setup
    if VISUALIZATION:
        fig, ax = plt.subplots()
        ax.set_title("Real-Time Data Stream with Anomaly Detection")
        ax.set_xlabel("Data Points")
        ax.set_ylabel("Value")
        ax.set_facecolor("black")
        fig.show()

    # Store anomalies to avoid recreating the legend
    anomaly_points = []

    while True:
        try:
            # Generate normal or abnormal data
            if random.random() <= OUTLIERS_GENERATION_PROBABILITY:
                X_test = np.random.uniform(low=-4, high=4, size=(1, 1))
            else:
                X = 0.3 * np.random.randn(1, 1)
                X_test = (X + np.random.choice(a=[2, -2], size=1, p=[0.5, 0.5]))

            X_test = np.round(X_test, 3).tolist()
            current_time = datetime.utcnow().isoformat()

            # Record for the incoming data
            record = {"id": _id, "data": X_test, "current_time": current_time}
            print(f"Incoming: {record}")

            # Load model
            try:
                model_path = os.path.abspath("isolation_forest.joblib")
                clf = load(model_path)
            except FileNotFoundError:
                logging.warning("Model file not found")
                print("Model file not available. Exiting.")
                break

            data = record['data']        
            data_ls.append(data[0][0])

            # Keep the size of data_ls optimized
            if len(data_ls) > MAX_DATA_POINTS:
                data_ls.pop(0)

            prediction = clf.predict(data)

            # Plot the real-time data
            if VISUALIZATION:
                ax.plot(range(len(data_ls)), data_ls, color='b', linewidth=0.8)
                ax.set_xlim(left=max(0, len(data_ls) - MAX_DATA_POINTS), right=len(data_ls))
                ax.grid(True, linestyle='--', alpha=0.5)

            # Check if an anomaly is detected
            if prediction[0] == -1:
                score = clf.score_samples(data)
                record["score"] = np.round(score, 3).tolist()
                print(f'Anomaly Detected: {record}')
                logging.info(f"Anomaly Detected: {record}")

                # Store anomaly points
                anomaly_points.append((len(data_ls) - 1, data_ls[-1]))

                # Update the anomaly plot
                if VISUALIZATION:
                    ax.scatter(*zip(*anomaly_points), color='r', s=50, label="Anomaly")

            _id += 1
            plt.pause(DELAY)

        except Exception as e:
            logging.error(f"Error during anomaly detection: {str(e)}")
            print(f"Error: {str(e)}")
            break
    
    plt.show()
    _id = 0

    # Visualization setup
    if VISUALIZATION:
        fig, ax = plt.subplots()
        ax.set_title("Real-Time Data Stream with Anomaly Detection")
        ax.set_xlabel("Data Points")
        ax.set_ylabel("Value")
        ax.set_facecolor("black")
        fig.show()

    while True:
        try:
            # Generate normal or abnormal data based on OUTLIERS_GENERATION_PROBABILITY
            if random.random() <= OUTLIERS_GENERATION_PROBABILITY:
                X_test = np.random.uniform(low=-4, high=4, size=(1, 1))
            else:
                X = 0.3 * np.random.randn(1, 1)
                X_test = (X + np.random.choice(a=[2, -2], size=1, p=[0.5, 0.5]))

            X_test = np.round(X_test, 3).tolist()
            current_time = datetime.utcnow().isoformat()

            # Creating a record for the incoming data
            record = {"id": _id, "data": X_test, "current_time": current_time}
            print(f"Incoming: {record}")

            # Loading the Isolation Forest model from the file
            try:
                model_path = os.path.abspath("isolation_forest.joblib")
                clf = load(model_path)
            except FileNotFoundError:
                logging.warning("Model file not found")
                print("Model file not available. Exiting.")
                break

            data = record['data']        
            data_ls.append(data[0][0])
            
            # Keeping data_ls size optimized for long-running processes
            if len(data_ls) > MAX_DATA_POINTS:
                data_ls.pop(0)

            prediction = clf.predict(data)

            # Updating visualization
            if VISUALIZATION:
                ax.plot(range(len(data_ls)), data_ls, color='b', linewidth=0.8)
                ax.set_xlim(left=max(0, len(data_ls)-MAX_DATA_POINTS), right=len(data_ls))
                ax.grid(True, linestyle='--', alpha=0.5)

            # Check if an anomaly is detected
            if prediction[0] == -1:
                score = clf.score_samples(data)
                record["score"] = np.round(score, 3).tolist()
                if VISUALIZATION:
                    ax.scatter(len(data_ls) - 1, data_ls[-1], color='r', s=50, label="Anomaly")
                    ax.legend(loc='upper right')
                logging.info(f"Anomaly Detected: {record}")
                print(f'Anomaly Detected: {record}')

            _id += 1
            plt.pause(DELAY)
        
        except Exception as e:
            logging.error(f"Error during anomaly detection: {str(e)}")
            print(f"Error: {str(e)}")
            break
    
    plt.show()
