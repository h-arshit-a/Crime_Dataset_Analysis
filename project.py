# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_style('whitegrid')

# --- Step 1: Load the Dataset ---
df = pd.read_csv('Crime_dataset.csv')  # Make sure the file exists
print("Initial Dataset Shape:", df.shape)

# --- Step 2: Data Cleaning ---
# Convert 'DATE OCC' to datetime
df['DATE OCC'] = pd.to_datetime(df['DATE OCC'],  errors='coerce')
#no format inferred as it will directly convert to datetime format and my format is yyyy-mm-dd

print(df['DATE OCC'].head(10))

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
df['Day_Night'] = np.where((df['Hour'] >= 6) & (df['Hour'] < 18), 'Day', 'Night')

print("Cleaned Dataset Shape:", df.shape)

# --- Step 3: Handling Missing Values ---
print("\nMissing Values:")
print(df.isnull().sum())

# Fill age nulls if any (precaution)
df['Vict Age'] = df['Vict Age'].fillna(df['Vict Age'].median())

# --- Step 4: Exploratory Data Analysis (EDA) ---

print("\nObjective 1-i: Evolution of Top 5 Crime Types Over Time:-")
# Filter top 5 crime types
top5_crimes = df['Crm Cd Desc'].value_counts().head(5).index
crime_time = df[df['Crm Cd Desc'].isin(top5_crimes)].groupby(['Year', 'Crm Cd Desc']).size().unstack()
# Make the plot bigger and wider
ax = crime_time.plot(kind='line', marker='o', figsize=(14, 7), linewidth=2)
# Titles and labels
plt.title("Top 5 Crime Types Over Years", fontsize=14,fontweight='bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Number of Crimes", fontsize=12)
# Move legend outside plot
plt.legend(title="Crime Type", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=10, title_fontsize=12)
# Add grid for clarity
plt.grid(True, linestyle='--', alpha=0.7)
# Ensure layout fits
plt.tight_layout()
# Show it
plt.show()
print("            ii: Yearly analysis of overall Crime Frequency:-")
crime_by_year = df['Year'].value_counts().sort_index()
# Plot Yearly Crime
plt.figure(figsize=(8, 5))
for x, y in crime_by_year.items():
    plt.text(x, y+2500, str(y), ha='center', va='bottom', fontsize=9)
crime_by_year.plot(kind='line', marker='o',color='red')
plt.title("Crimes per Year",fontweight='bold')
plt.xlabel("Year",fontsize=12)
plt.ylabel("Number of Crimes",fontsize=12)
plt.grid(True)
plt.show()
# Objective 2: Crime by Hour of Day and Day of Week

print("\nObjective 2: Crime by Hour of Day and Day of Week")
hourly_crime = df['Hour'].value_counts().sort_index()
weekday_crime = df['Day_of_Week'].value_counts().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)

# Plot Crimes by Hour as Histogram
plt.figure(figsize=(10, 6))
plt.hist(df['Hour'], bins=24, color='green', edgecolor='black')
plt.title("Hourly Crime Analysis",fontweight='bold')
plt.xlabel("Time (in 24-hour format)",fontsize=12)
plt.ylabel("Crime Count",fontsize=12)
plt.xticks(range(0, 24))
plt.grid(True)
plt.show()

# Plot Crimes by Day
plt.figure(figsize=(10, 6))
weekday_crime.plot(kind='barh', color='purple')
plt.title("Crimes by Day of Week",fontweight='bold')
plt.ylabel("Days of a Week",fontsize=12)
plt.xlabel("Crime Count",fontsize=12)
plt.xticks(rotation=0)
plt.show()

# Objective 3: Area-wise and Descent-wise Crime Distribution
print("\nObjective 3: Crime Distribution by Area and Victim Descent")
area_crime = df['AREA NAME'].value_counts()
descent_crime = df['Vict Descent'].value_counts()
plt.figure(figsize=(12, 8))
norm = plt.Normalize(vmin=area_crime.min(), vmax=area_crime.max())
colors = plt.cm.Reds(norm(area_crime))
area_crime.plot(kind='bar', color=colors)
plt.title("Crimes by Area",fontweight='bold')
plt.ylabel("Number of Crimes",fontsize=12)
plt.xticks(rotation=90)
# Annotate top 3 areas
top_areas = area_crime.nlargest(3)
for area, count in top_areas.items():
    plt.text(area_crime.index.tolist().index(area), count + 5, f'{count}', 
             ha='center')
plt.show()
# Plot Descent-wise Histogram
# Plot Descent-wise Bar Plot with Full Descent Names
descent_mapping = {
    'A': 'Other Asian', 'B': 'Black', 'C': 'Chinese', 'D': 'Korean', 
    'F': 'Filipino', 'G': 'Guamanian', 'H': 'Hispanic', 
    'I': 'American Indian', 'J': 'Japanese', 
    'K': 'Vietnamese', 'L': 'Laotian', 'O': 'Other', 
    'P': 'Pacific Islander', 'S': 'Samoan', 'U': 'Hawaiian', 
    'V': 'Vietnamese', 'W': 'White', 'X': 'Unknown', 'Z': 'Asian Indian'
}
descent_crime_full = descent_crime.rename(index=descent_mapping)
plt.figure(figsize=(12, 8))
descent_crime_full.plot(kind='bar', color='pink', edgecolor='black')
plt.title("Crimes by Victim Descent",fontweight='bold')
plt.xlabel("Victim Descent(Race)",fontsize=12)
plt.ylabel("Frequency",fontsize=12)
plt.xticks(rotation=90)
plt.grid(True)
plt.show()



# Objective 4: Victim Demographics for Top 5 Crimes and Victim Age vs Crime Type Heatmap (Replaces boxplot for top 5 crimes)
print("\nObjective 4-i: Victim Age vs Crime Type Heatmap")
top5_crimes = df['Crm Cd Desc'].value_counts().head(5).index
# Mapping for shortened crime type names
crime_short_names = {
    'ASSAULT WITH DEADLY WEAPON, AGGRAVATED ASSAULT': 'Assault DW',
    'BATTERY - SIMPLE ASSAULT': 'Assault',
    'BURGLARY FROM VEHICLE': 'Vhl-brglry',
    'THEFT OF IDENTITY': 'ID Theft',
    'THEFT PLAIN - PETTY ($950 & UNDER)': 'Petty Tft'  # Adjust based on your top 5 crimes
}
# Ensure the mapping covers the top 5 crimes, using original names as fallback
top5_short = [crime_short_names.get(crime, crime) for crime in top5_crimes]

age_crime = pd.crosstab(pd.cut(df[df['Crm Cd Desc'].isin(top5_crimes)]['Vict Age'], bins=10), 
                        df[df['Crm Cd Desc'].isin(top5_crimes)]['Crm Cd Desc'])
# Rename columns with shortened names
age_crime.columns = top5_short

plt.figure(figsize=(10, 8))
sns.heatmap(age_crime, annot=True, cmap='YlOrRd', fmt='d')
plt.title("Victim Age vs Top 5 Crimes",fontweight='bold')
plt.xlabel("Crime Type",fontsize=12)
plt.ylabel("Victim Age Range",fontsize=12)
plt.xticks(rotation=0)  # Adjusted rotation for better fit
plt.show()

print("            ii: Gender-wise crime distribution for top 5 crimes")
top5_df = df[df['Crm Cd Desc'].isin(top5_crimes)]

gender_dist = top5_df['Vict Sex'].value_counts()
# Plotting gender distribution for top 5 crimes
plt.figure(figsize=(6, 6))  # Slightly larger for better visibility
plt.pie(gender_dist, labels=gender_dist.index, autopct='%1.1f%%', 
        colors=plt.cm.Set2.colors, startangle=90, shadow=True, 
        wedgeprops={'edgecolor': 'white', 'linewidth': 1})
plt.title("Gender-wise Crime Distribution for Top 5 Crimes", fontsize=12, fontweight='bold')
plt.legend(labels=['Male (M)', 'Female (F)'], loc='best', title="Gender Codes", 
           fontsize=10, title_fontsize=12)
plt.tight_layout()
plt.show()

# --- Hypothesis Testing: Is Day/Night Crime Ratio 50/50? ---
print("\n Objective 5: Let the Hypothesis be: Daytime crimes = 50% of total")
total = len(df)
day_crimes = len(df[df['Day_Night'] == 'Day'])
p_hat = day_crimes / total
p_0 = 0.5
se = np.sqrt(p_0 * (1 - p_0) / total)
z = (p_hat - p_0) / se
p_val = 2 * (1 - abs(z))  # Rough approximation
print(f"Observed Daytime Proportion: {p_hat:.3f}")
print(f"Z-Score: {z:.3f}")
print(f"Approximate P-Value: {p_val:.3f}")
if p_val < 0.05:
    print("Conclusion: Significant difference from 50%.")#the actual percentage of crimes that occurred during the day (e.g., 0.45 means 45% of crimes were daytime).
else:
    print("Conclusion: No significant difference from 50%.")

# --- Outlier Detection on Victim Age ---
print("\nObjective 6: Outlier Detection on Victim Age")

ages = df['Vict Age']
Q1 = np.percentile(ages, 25)
Q3 = np.percentile(ages, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = ages[(ages < lower_bound) | (ages > upper_bound)]

print(f"Q1 = {Q1}, Q3 = {Q3}, IQR = {IQR}")
print(f"Lower Bound = {lower_bound}, Upper Bound = {upper_bound}")
print(f"Number of Outliers Detected: {len(outliers)}")

# Boxplot with enhanced visualization
plt.figure(figsize=(8, 6))  # Increased figure size for clarity
plt.boxplot(ages, patch_artist=True, 
            boxprops=dict(facecolor='lightblue', color='blue'),
            whiskerprops=dict(color='blue'),
            capprops=dict(color='blue'),
            medianprops=dict(color='green', linewidth=2),
            flierprops=dict(marker='o', markerfacecolor='red', markersize=8, linestyle='none'))
plt.title("Victim Age Outlier Detection", fontsize=12, fontweight='bold')
plt.ylabel("Victim Age", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
# Add legend for green (median) and red (outliers)
plt.legend([plt.Line2D([10], [10], color='green', lw=2), 
            plt.Line2D([10], [10], marker='o', color='black', markerfacecolor='red', markersize=8, lw=0)],
           ['Median', 'Outliers'], loc='best')
plt.show()

# --- Correlation & Covariance Analysis ---
print("\n Objective 7: Correlation and Covariance (Numeric Features)")
numeric_cols = ['Vict Age', 'LAT', 'LON', 'Hour']
correlation = df[numeric_cols].corr()
covariance = df[numeric_cols].cov()
#There is very little meaningful correlation in this heatmap — most values are close to 0, which indicates weak or no linear relationship between your variables.
# LAT & LON	-0.58	Moderate negative correlation — This is the only notable relationship in the entire heatmap.
# All other pairs	~0 (e.g., -0.01, 0.036, 0.0072)	No correlation
print("\nCorrelation Matrix:")
print(correlation)
print("\nCovariance Matrix:")
print(covariance)

# Plot correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(correlation, annot=True, cmap='YlGnBu',fmt=".2f",square= True)
plt.title("Correlation Heatmap",fontweight='bold')
plt.show()

print("\nObjective 8: Day vs. Night Crime Distribution")
day_night_counts = df['Day_Night'].value_counts()
# Create pie chart
colors = sns.color_palette("rocket", n_colors=2)  # You can try "rocket", "flare", etc.

# Create pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    day_night_counts.values,
    labels=day_night_counts.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors
)
plt.title("Crimes During Day vs Night",fontweight='bold')
plt.tight_layout()
plt.show()
#end
