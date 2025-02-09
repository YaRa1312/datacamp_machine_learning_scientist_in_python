# You have been asked to support a team of researchers who have been collecting data about penguins in Antartica! The data is available in csv-Format as penguins.csv

# Origin of this data : Data were collected and made available by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.

# The dataset consists of 5 columns.

# Column	Description
# culmen_length_mm	culmen length (mm)
# culmen_depth_mm	culmen depth (mm)
# flipper_length_mm	flipper length (mm)
# body_mass_g	body mass (g)
# sex	penguin sex
# Unfortunately, they have not been able to record the species of penguin, but they know that there are at least three species that are native to the region: Adelie, Chinstrap, and Gentoo. Your task is to apply your data science skills to help them identify groups in the dataset!

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
