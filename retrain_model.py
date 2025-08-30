import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

# Encode Gender
if 'Gender' in df.columns:
    df['Gender_code'] = df['Gender'].map({'Male': 1, 'Female': 0})
else:
    df['Gender_code'] = 0

# Detect correct income and score columns
income_col = [c for c in df.columns if 'Income' in c][0]
score_col = [c for c in df.columns if 'Spending' in c][0]

# Prepare features
X = df[['Gender_code', 'Age', income_col, score_col]].dropna()

# Train KMeans model
model = KMeans(n_clusters=4, random_state=42)
clusters = model.fit_predict(X)

# Save model
joblib.dump(model, "KMeans_model.pkl")
print("✅ Model retrained and saved as KMeans_model.pkl")

# Add cluster labels to dataframe and save as cleaned.csv
df['clusters'] = clusters
df.to_csv("cleaned.csv", index=False)
print("✅ Cleaned dataset saved as cleaned.csv")
