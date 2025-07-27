import pandas as pd

# Load dataset
df = pd.read_csv('uber.csv')

# Ensure timestamp columns are in datetime format
if 'pickup_datetime' in df.columns:
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')

    # Feature engineering
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day'] = df['pickup_datetime'].dt.day
    df['month'] = df['pickup_datetime'].dt.month
    df['day_of_week'] = df['pickup_datetime'].dt.day_name()

    # Create peak/off-peak indicator (example: peak hours 7-9 AM, 5-7 PM)
    df['peak'] = df['hour'].apply(lambda x: 1 if x in [7,8,9,17,18,19] else 0)
else:
    print("pickup_datetime column not found. Feature engineering skipped.")

# Handle categorical variables (if any)
if 'store_and_fwd_flag' in df.columns:
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'Y': 1, 'N': 0})

# Save the enhanced dataset
df.to_csv('uber_enhanced.csv', index=False)
print("Enhanced dataset saved as uber_enhanced.csv")

