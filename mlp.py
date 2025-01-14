import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('./IR data/consolidated_spectrum_cleaned.CSV')

concentrations = data.columns[2:]  

# Features are the Np array of IR reading 
features = data.iloc[:, 2:].values.T  #  Reshape the data to have rows of IR readings and corresponding concentrations
# Target (concentrations) is the column headers
targets = data.columns[2:].values


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Initialize and train the MLP regressor
mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50, 50, 50, 50), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)


# Evaluate the model
predictions = mlp.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")


"""
# Define a function for predicting concentration from new IR readings
def predict_concentration(ir_readings):
    ir_readings = np.array(ir_readings).reshape(1, -1)
    predicted_concentration = mlp.predict(ir_readings)
    return predicted_concentration

# Load test data
data = pd.read_csv('csvname', header=False)
data = data.rename(columns={data.columns[0]: "Wavelength"})  # Rename the first column
new_sample = data.iloc[:, 1:].values.T  # Transpose the IR data for proper mapping
predicted = predict_concentration(new_sample)
print(f"Predicted Concentration: {predicted[0]}")
"""