import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Simulate Logistics Delivery Dataset
data = {
    'delivery_id': range(101, 111),
    'latitude': [12.9716, 12.9352, 12.9279, 12.9915, 12.9784, 12.9141, 12.9611, 12.9856, 12.9344, 12.9592],
    'longitude': [77.5946, 77.6245, 77.6271, 77.5532, 77.6408, 77.6109, 77.5735, 77.6012, 77.6101, 77.6974],
    'package_weight_kg': [2.5, 5.0, 1.2, 10.4, 3.1, 7.8, 4.2, 0.8, 6.5, 3.0],
    'distance_km': [5.2, 12.1, 8.4, 18.0, 6.7, 14.3, 9.1, 4.0, 11.5, 16.2],
    'delivery_time_min': [22, 45, 30, 65, 28, 52, 36, 18, 44, 58]
}
df = pd.DataFrame(data)

# 2. Cluster Delivery Locations into 2 Route Zones
coords = df[['latitude', 'longitude']]
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['delivery_zone'] = kmeans.fit_predict(coords)

# 3. Train a Predictive Model for Estimated Delivery Time (ETA)
X = df[['package_weight_kg', 'distance_km', 'delivery_zone']]
y = df['delivery_time_min']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# Example Prediction
sample_trip = pd.DataFrame({'package_weight_kg': [3.5], 'distance_km': [7.0], 'delivery_zone': [0]})
predicted_eta = model.predict(sample_trip)
print(f"Predicted ETA: {predicted_eta[0]:.2f} minutes")
