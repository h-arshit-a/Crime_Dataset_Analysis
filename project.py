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

# --- Step 4: Exploratory Data Analysis (EDA) ---

# Objective 1: Crime Frequency by Year Evolution of Top 5 Crime Types Over Time
print("\nObjective 7: Evolution of Top 5 Crime Types Over Time")
top5_crimes = df['Crm Cd Desc'].value_counts().head(5).index
crime_time = df[df['Crm Cd Desc'].isin(top5_crimes)].groupby(['Year', 'Crm Cd Desc']).size().unstack()

plt.figure(figsize=(10, 6))
crime_time.plot(kind='line', marker='o')
plt.title("Top 5 Crime Types Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Crimes")
plt.legend(title="Crime Type")
plt.grid(True)
plt.show()
print("\nObjective 1: Crime Frequency by Year and Season")
crime_by_year = df['Year'].value_counts().sort_index()
crime_by_season = df['Season'].value_counts()

# Plot Yearly Crime
plt.figure(figsize=(8, 5))
crime_by_year.plot(kind='line', marker='o')
plt.title("Crimes per Year")
plt.xlabel("Year")
plt.ylabel("Number of Crimes")
plt.grid(True)
plt.show()
# Objective 2: Crime by Hour of Day and Day of Week
print("\nObjective 2: Crime by Hour and Day of Week")
hourly_crime = df['Hour'].value_counts().sort_index()
weekday_crime = df['Day_of_Week'].value_counts().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

# Plot Crimes by Hour
plt.figure(figsize=(10, 4))
hourly_crime.plot(kind='bar', color='orange')
plt.title("Crimes by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Crime Count")
plt.xticks(rotation=0)
plt.show()

# Plot Crimes by Day
plt.figure(figsize=(8, 4))
weekday_crime.plot(kind='bar', color='purple')
plt.title("Crimes by Day of Week")
plt.xlabel("Day")
plt.ylabel("Crime Count")
plt.xticks(rotation=45)
plt.show()















