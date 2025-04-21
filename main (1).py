# Importing necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['Species'] = iris.target
df['Species'] = df['Species'].replace({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Label Encoding
label_encoder = LabelEncoder()
df['Species_encoded'] = label_encoder.fit_transform(df['Species'])

# Features and Target
X = df.drop(['Species', 'Species_encoded'], axis=1)
y = df['Species_encoded']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Display shape and sample
print("Encoded Classes:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Sample y_train:", y_train[:5])
print("Sample y_test:", y_test[:5])
