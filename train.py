# train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Paths
DATA_PATH = "data/relationship_sm.csv"
MODEL_PATH = "models/decision_tree_model.pkl"

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# # Rename columns for consistency with app.py
# rename_map = {
#     "Daily_Instagram_Hours": "Instagram_Hours",
#     "Male_Friends_Count": "Male_Friends",
#     "Female_Friends_Count": "Female_Friends",
#     "Party_Frequency_Monthly": "Party_Frequency",
#     "Late_Night_Calls_Per_Week": "Late_Night_Talks",
#     "Gym_Sessions_Per_Week": "Gym_Frequency",
#     "Gaming_Hours_Weekly": "Gaming_Hours",
#     "Shopping_Trips_Monthly": "Shopping_Frequency",
#     "Travel_Frequency_Yearly": "Travel_Frequency",
#     "Music_Listening_Hours_Daily": "Music_Hours",
#     "Gender": "Gender"
# }
# df = df.rename(columns=rename_map)

# # Features and target
# X = df.drop(columns=["Status"])  # Status = 0 Single, 1 Committed
# y = df["Status"]

selected_features = [
    "Num_Male_Friends", "Num_Female_Friends", "Daily_Instagram_Hours",
    "Party_Frequency", "Late_Night_Talks_Per_Week", "Study_Hours",
    "Messaging_Apps_Used", "Texts_Per_Day", "Parents_Strictness_Level",
    "Coffee_Shop_Visits", "Gender"
]

X = df[selected_features]
y = df["Status"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Decision Tree
model = DecisionTreeClassifier(max_depth=6, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
