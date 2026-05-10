import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load processed dataset
df = pd.read_csv("../data/processed/creditcard_with_hour.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale Amount and Time features
scaler = StandardScaler()

X_train[["Amount", "Time", "Hour"]] = scaler.fit_transform(
    X_train[["Amount", "Time", "Hour"]]
)

X_test[["Amount", "Time", "Hour"]] = scaler.transform(
    X_test[["Amount", "Time", "Hour"]]
)

# Save processed datasets
X_train.to_csv("../data/processed/X_train.csv", index=False)
X_test.to_csv("../data/processed/X_test.csv", index=False)

y_train.to_csv("../data/processed/y_train.csv", index=False)
y_test.to_csv("../data/processed/y_test.csv", index=False)

print("Preprocessing completed successfully.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
