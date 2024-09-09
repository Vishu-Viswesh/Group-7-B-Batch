# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Load dataset - Golub et al.
df_train = pd.read_csv("/content/data_set_ALL_AML_train.csv")
df_test = pd.read_csv("/content/data_set_ALL_AML_independent.csv")
df_actual = pd.read_csv("/content/actual.csv")

# Remove non-numeric or irrelevant columns
columns_to_drop = ['Gene Description', 'Gene Accession Number'] + [col for col in df_train.columns if 'call' in col]
df_train_cleaned = df_train.drop(columns=columns_to_drop)

# Select a target column
print(df_train_cleaned.columns)

# Take column '1' as the target column for prediction 
X = df_train_cleaned.drop(columns=['1'])  # Columns are features
y = df_train_cleaned['1']  # This is the target for prediction

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for better performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the ANN model using tensorflow->Keras->Dense
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Expression Level")
plt.ylabel("Predicted Expression Level")
plt.title("Predicted vs Actual Gene Expression Levels")
plt.show()

# Plot training & validation loss over epochs
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.title("Training and Validation Loss over Epochs")
plt.show()
