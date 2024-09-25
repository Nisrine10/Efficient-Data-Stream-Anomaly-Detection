from model_prod import model
from anomaly import anomaly_dect

if __name__ == '__main__':
    model()  # Train the model first
    anomaly_dect()  # Start real-time anomaly detection
