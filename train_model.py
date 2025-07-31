import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("heart.csv")

# Preprocessing (if 'thal' or 'cp' has strings, convert them — else skip)
# If needed, uncomment this:
# df['thal'] = df['thal'].replace({'normal':3, 'fixed':6, 'reversible':7})
# df['sex'] = df['sex'].replace({'male':1, 'female':0})

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model as model.pkl
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as model.pkl")
