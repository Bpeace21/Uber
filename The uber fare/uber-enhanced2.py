import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# Haversine formula to calculate distance in km
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    km = 6371 * c  # Earth radius in km
    return km

# Load dataset
df = pd.read_csv('uber_enhanced2.csv')

# Ensure datetime format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# Calculate distance (km) between pickup and dropoff coordinates
df['distance_km'] = df.apply(
    lambda row: haversine(
        row['pickup_longitude'], row['pickup_latitude'],
        row['dropoff_longitude'], row['dropoff_latitude']
    ),
    axis=1
)

# Estimate duration (assuming average speed of 20 mph ≈ 32.19 km/h)
AVG_SPEED_KMPH = 32.19  # Adjust based on traffic conditions
df['estimated_duration_hours'] = df['distance_km'] / AVG_SPEED_KMPH
df['estimated_duration_seconds'] = df['estimated_duration_hours'] * 3600

# Generate dropoff_datetime
df['dropoff_datetime'] = df['pickup_datetime'] + pd.to_timedelta(df['estimated_duration_seconds'], unit='s')

# Convert duration to minutes for readability
df['ride_duration_minutes'] = df['estimated_duration_seconds'] / 60

# Feature engineering (optional)
df['hour'] = df['pickup_datetime'].dt.hour
df['day'] = df['pickup_datetime'].dt.day
df['month'] = df['pickup_datetime'].dt.month
df['day_of_week'] = df['pickup_datetime'].dt.day_name()

# Peak hours indicator (7-9 AM, 5-7 PM)
df['peak'] = df['hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)

# Save enhanced dataset
df.to_csv('uber_enhanced2.csv', index=False)
print("Enhanced dataset saved as 'uber_enhanced2.csv'")