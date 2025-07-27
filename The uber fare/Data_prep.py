import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import os
import sys

# Load dataset
csv_file = 'uber.csv'
if not os.path.isfile(csv_file):
    print(f"Error: '{csv_file}' not found in the current directory.")
    sys.exit(1)

df = pd.read_csv(csv_file)

# Convert pickup_datetime to datetime format
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

# Dataset overview
print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())
print("\nData types:\n", df.dtypes)
print("\nFirst 5 rows:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# Drop rows with missing values
df_cleaned = df.dropna()
df_cleaned.to_csv("uber_cleaned.csv", index=False)
print("\nCleaned dataset saved as 'uber_cleaned.csv'")
# Haversine distance function
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return 6371 * c  # in km

# Compute distance
df_cleaned['distance_km'] = df_cleaned.apply(
    lambda row: haversine(
        row['pickup_longitude'], row['pickup_latitude'],
        row['dropoff_longitude'], row['dropoff_latitude']
    ), axis=1
)

# Estimate duration (assuming avg speed ~32.19 km/h)
AVG_SPEED_KMPH = 32.19
df_cleaned['estimated_duration_hours'] = df_cleaned['distance_km'] / AVG_SPEED_KMPH
df_cleaned['estimated_duration_seconds'] = df_cleaned['estimated_duration_hours'] * 3600
df_cleaned['dropoff_datetime'] = df_cleaned['pickup_datetime'] + pd.to_timedelta(df_cleaned['estimated_duration_seconds'], unit='s')
df_cleaned['ride_duration_minutes'] = df_cleaned['estimated_duration_seconds'] / 60

# Time-based features
df_cleaned['hour'] = df_cleaned['pickup_datetime'].dt.hour
df_cleaned['day'] = df_cleaned['pickup_datetime'].dt.day
df_cleaned['month'] = df_cleaned['pickup_datetime'].dt.month
df_cleaned['day_of_week'] = df_cleaned['pickup_datetime'].dt.day_name()
df_cleaned['peak'] = df_cleaned['hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)

# Encode categorical variable
if 'store_and_fwd_flag' in df_cleaned.columns:
    df_cleaned['store_and_fwd_flag'] = df_cleaned['store_and_fwd_flag'].map({'Y': 1, 'N': 0})

# Save enhanced dataset
df_cleaned.to_csv('uber_enhanced.csv', index=False)
print("Enhanced dataset saved as uber_enhanced.csv")

# Descriptive statistics
print("\nDescriptive statistics:\n", df_cleaned.describe())
print("Mean Fare:", df_cleaned['fare_amount'].mean())
print("Median Fare:", df_cleaned['fare_amount'].median())
print("Mode Fare:", df_cleaned['fare_amount'].mode()[0])
print("Fare Std Dev:", df_cleaned['fare_amount'].std())
print("Fare Quartiles:\n", df_cleaned['fare_amount'].quantile([0.25, 0.5, 0.75]))

# Outlier detection using IQR
Q1 = df_cleaned['fare_amount'].quantile(0.25)
Q3 = df_cleaned['fare_amount'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df_cleaned[(df_cleaned['fare_amount'] < lower) | (df_cleaned['fare_amount'] > upper)]
print(f"Number of outliers in fare: {len(outliers)}")
# Fare Distribution
plt.figure(figsize=(8,5))
sns.histplot(df_cleaned['fare_amount'], bins=50, kde=True)
plt.title('Fare Amount Distribution')
plt.xlabel('Fare Amount')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("fare_distribution.png")
plt.show()

# Fare vs Distance
plt.figure(figsize=(8,5))
sns.scatterplot(x='distance_km', y='fare_amount', data=df_cleaned)
plt.title('Fare vs Distance')
plt.xlabel('Distance (km)')
plt.ylabel('Fare Amount')
plt.tight_layout()
plt.savefig("fare_vs_distance.png")
plt.show()

# Fare vs Hour
plt.figure(figsize=(8,5))
sns.boxplot(x='hour', y='fare_amount', data=df_cleaned)
plt.title('Fare vs Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Fare Amount')
plt.tight_layout()
plt.savefig("fare_vs_hour.png")
plt.show()
