# Measuring essential soil metrics such as nitrogen, phosphorous, potassium levels, and pH value is an important aspect of assessing soil condition. However, it can be an expensive and time-consuming process, which can cause farmers to prioritize which metrics to measure based on their budget constraints.

# Farmers have various options when it comes to deciding which crop to plant each season. Their primary objective is to maximize the yield of their crops, taking into account different factors. One crucial factor that affects crop growth is the condition of the soil in the field, which can be assessed by measuring basic elements such as nitrogen and potassium levels. Each crop has an ideal soil condition that ensures optimal growth and maximum yield.

# A farmer reached out to you as a machine learning expert for assistance in selecting the best crop for his field. They've provided you with a dataset called soil_measures.csv, which contains:

# "N": Nitrogen content ratio in the soil
# "P": Phosphorous content ratio in the soil
# "K": Potassium content ratio in the soil
# "pH" value of the soil
# "crop": categorical values that contain various crops (target variable).
# Each row in this dataset represents various measures of the soil in a particular field. Based on these measurements, the crop specified in the "crop" column is the optimal choice for that field.

# In this project, you will build multi-class classification models to predict the type of "crop" and identify the single most importance feature for predictive performance.

# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")
# print(list(crops.columns))

# Write your code here
feature_names = ["N", "P", "K", "ph"]

# Ensure the column names are correctly referenced
crops.columns = crops.columns.str.strip()

feature_scores = {}

for feature in feature_names:
    X = crops[[feature]]
    y = crops["crop"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    y_pred = logreg.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    
    feature_scores[feature] = accuracy
    
best_feature = max(feature_scores, key=feature_scores.get)
best_predictive_feature = {best_feature: feature_scores[best_feature]}

print("Best predictive feature: ", best_predictive_feature)
