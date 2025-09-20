# Question: In two-vehicle collisions, do crashes where both drivers are assigned
# the same contributing factor (eg. both 'Driver Inattention', or both 'Failure to Yield')
# result in more severe outcomes than crashes where the drivers are assigned
# different factors (eg. one 'Driver inattention' and one 'Unspecified')?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load Data ---
df = pd.read_csv('Motor_Vehicle_Collisions_Crashes.csv', parse_dates=['CRASH DATE'])
print("\nAmount of rows: ", len(df))

# --- Clean contributing factors ---
def norm_factor(x):
    if pd.isna(x) or str(x).strip() == "":
        return "Unspecified"
    return str(x).strip().title()

for col in ['CONTRIBUTING FACTOR VEHICLE 1', 'CONTRIBUTING FACTOR VEHICLE 2']:
    df[col] = df[col].apply(norm_factor)

# --- Filter: two-vehicle collisions ---
mask_two_vehicle = (
    df['VEHICLE TYPE CODE 1'].notna() &
    df['VEHICLE TYPE CODE 2'].notna() &
    df['VEHICLE TYPE CODE 3'].isna() & df['VEHICLE TYPE CODE 4'].isna() & df['VEHICLE TYPE CODE 5'].isna()
)
df2 = df[mask_two_vehicle].copy()
print("Two-vehicle collisions:", len(df2))

# --- Severeity Metrics ---
df2['total_injuries'] = df2['NUMBER OF PERSONS INJURED'].fillna(0).astype(int)
df2['total_killed'] = df2['NUMBER OF PERSONS KILLED'].fillna(0).astype(int)
df2['severity_count'] = df2['total_injuries'] + df2['total_killed']
df2['severe_binary'] = ((df2['total_killed'] >= 1) | (df2['total_injuries'] >= 2)).astype(int)

# --- Matching Factors ---
df2['factors_match'] = (df2['CONTRIBUTING FACTOR VEHICLE 1'] == df2['CONTRIBUTING FACTOR VEHICLE 2']).astype(int)
df2['factor_pair'] = df2['CONTRIBUTING FACTOR VEHICLE 1'] + " || " + df2['CONTRIBUTING FACTOR VEHICLE 2']

# --- Summary ---
summary = df2.groupby('factors_match').agg(
    n_crashes=('COLLISION_ID','count'),
    mean_severity=('severity_count','mean'),
    pct_severe=('severe_binary','mean')
).reset_index()
summary['pct_severe'] *= 100
print(summary)

# --- Visualizations ---
# Boxplot of severity
plt.figure(figsize=(7,5))
sns.boxplot(x='factors_match', y='severity_count', data=df2, hue='factors_match', palette="Set2")
plt.xticks([0,1], ['Non-Matching', 'Matching'])
plt.ylabel('Total Persons Injured + Killed')
plt.title('Crash Severity by Matching vs. Non-Matching Factors')

# Bar plot of percent severe
plt.figure(figsize=(6,4))
sns.barplot(x='factors_match', y='severe_binary', data=df2, estimator=np.mean, palette="Set2")
plt.xticks([0,1], ['Non-Matching', 'Matching'])
plt.ylabel('% Severe Crashes')
plt.title('Percent Severe by Matching vs. Non-Matching Factors')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

# Top factor pairs
top_pairs = df2['factor_pair'].value_counts().head(15)
print("Top factor pairs:\n", top_pairs)

plt.figure(figsize=(8,6))
top_pairs.plot(kind='barh', color='skyblue')
plt.xlabel("Number of Collisions")
plt.title("Top 15 Factor Pairs in Two-Vehicle Collisions")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.show()