# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_style('whitegrid')

# --- Step 1: Load the Dataset ---
df = pd.read_csv('crime_dataset.csv')  # Make sure the file exists
print("Initial Dataset Shape:", df.shape)

# --- Step 2: Data Cleaning ---
# Convert 'DATE OCC' to datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'], errors='coerce')

# Format 'TIME OCC' and extract hour
df['TIME OCC'] = df['TIME OCC'].astype(str).str.zfill(4)
df['Hour'] = df['TIME OCC'].str[:2].astype(int)

# Remove invalid coordinates
df = df[(df['LAT'] != 0) & (df['LON'] != 0)]

# Keep valid gender values only
df = df[df['Vict Sex'].isin(['M', 'F'])]

# Remove negative or zero age
df = df[df['Vict Age'] > 0]

# Drop nulls in important columns
df.dropna(subset=['AREA NAME', 'Crm Cd Desc', 'DATE OCC', 'Vict Descent'], inplace=True)

# Create derived columns
df['Month'] = df['DATE OCC'].dt.month
df['Year'] = df['DATE OCC'].dt.year
df['Day_of_Week'] = df['DATE OCC'].dt.day_name()
df['Season'] = pd.cut(df['Month'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'], include_lowest=True)
df['Day_Night'] = np.where((df['Hour'] >= 6) & (df['Hour'] < 18), 'Day', 'Night')

print("Cleaned Dataset Shape:", df.shape)

# --- Step 3: Handling Missing Values ---
print("\nMissing Values:")
print(df.isnull().sum())

# Fill age nulls if any (precaution)
df['Vict Age'] = df['Vict Age'].fillna(df['Vict Age'].median())















