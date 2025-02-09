# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("penguins.csv")
penguins_df.head()

# My code
# Check basic statistics
# print(penguins_df.describe())

# Copying the columns to a new var to have an access to it later
penguins_original = penguins_df.copy()

# Dropping the 'sex' column as a non-numerical
penguins_numeric = penguins_df.drop(columns=['sex'])

# Standardizing the numeric columns
scaler = StandardScaler()
penguins_scaled = scaler.fit_transform(penguins_numeric)

# Converting back to DataFrame
penguins_scaled_df = pd.DataFrame(penguins_scaled, columns=penguins_numeric.columns)

# Setting K to 3 (because we have 3 species)
kmeans = KMeans(n_clusters=3, random_state=42)

# Fitting and predicting clusters
penguins_scaled_df['Cluster'] = kmeans.fit_predict(penguins_scaled_df)

# Adding cluster labels to original DataFrame
penguins_df['Cluster'] = penguins_scaled_df['Cluster']

# Computing mean values for each cluster
stat_penguins = penguins_df.groupby("Cluster").mean()

# Displaying the final DataFrame
print(stat_penguins)
