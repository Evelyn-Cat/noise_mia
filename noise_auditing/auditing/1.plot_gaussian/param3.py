from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor  # Example model; you can change this
import numpy as np
import pandas as pd

q = []
clip = []
k = [0.5076, 6.6256, 8.8894, 5.6432, 7.5653, 9.6156, 9.103, 9.4447, 9.6583, 9.5729, 9.402, 9.1884, 8.9322, 9.6156, 9.1884, 9.9146, 9.3593, 9.9573, 9.2739, 9.7864, 9.1884, 9.6583, 8.8894, 9.3166, 9.7864, 8.9749, 9.3166, 9.701, 8.8894, 9.2312, 9.5302, 9.8719, 9.0176, 9.3166, 9.5729, 9.8719, 8.9749, 9.2312, 9.4874, 9.7437, 10.0, 9.0603, 9.3166, 9.5302, 9.7437, 9.9573]
theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
acc = [73.5, 73.66, 71.12, 73.19, 72.50999999999999, 72.89999999999999, 73.35000000000001, 72.33000000000001, 73.02, 71.69, 72.72, 72.87, 74.18, 72.19, 73.41, 73.09, 72.85000000000001, 72.18, 73.11999999999999, 72.11, 72.65, 73.0, 72.77, 72.97, 72.86, 73.94, 72.16, 73.02, 72.92999999999999, 73.81, 72.55, 73.47, 72.66, 73.28, 70.81, 72.69, 72.46000000000001, 72.59, 73.63, 73.29, 72.67, 73.2, 72.89999999999999, 72.19, 73.9, 73.19]
mia = [0.08434114843375096, 0.08231704248921375, 0.062880898039201, 0.08810244822154573, 0.0769610411361284, 0.08369755902156667, 0.07436508195530674, 0.0822249402556695, 0.07166928669035968, 0.06849941642469226, 0.07027205298533169, 0.07176236622689079, 0.07991962317587344, 0.07705362944229147, 0.08010424424166136, 0.0829615207143473, 0.07352923328811553, 0.07807153542015544, 0.07334339422420959, 0.07455073126429625, 0.07529298403543135, 0.07408654335269817, 0.06859279146561167, 0.07436508195530674, 0.06812582904760044, 0.07631268284916351, 0.07529298403543135, 0.07547846117590556, 0.07510747248680548, 0.0822249402556695, 0.07455073126429625, 0.07677583880204963, 0.0764053312024053, 0.0693394780786129, 0.0701788346212465, 0.07760897932698288, 0.07436508195530674, 0.07705362944229147, 0.0820407103362094, 0.08010424424166136, 0.06896620446472282, 0.08194858264716733, 0.08038111194681256, 0.06961934188005707, 0.08837711061741292, 0.08148781684626792]

# Sample Data (Replace this with your actual dataset)
# For example, assume we have some initial values for a, m, k, and theta
data = {
    'accuracy': acc,  # Replace with real data
    'mia':mia,       # Replace with real data
    'k': k,                  # Replace with real data
    'theta': theta             # Replace with real data
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define the regression function
def train_and_predict_k_theta(data_df, accuracy, mia):
    """
    Trains a multi-output regression model on given data to predict `k` and `theta`
    based on `accuracy` and `mia`, and returns predictions for input accuracy and mia values.
    
    Parameters:
    data_df (pd.DataFrame): Data containing columns `accuracy`, `mia`, `k`, `theta`.
    accuracy (float): Input value for accuracy to predict `k` and `theta`.
    mia (float): Input value for mia to predict `k` and `theta`.
    
    Returns:
    tuple: Predicted values for `k` and `theta`.
    """
    # Features and targets from the DataFrame
    X = data_df[['accuracy', 'mia']]
    y = data_df[['k', 'theta']]
    
    # Model setup
    model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
    
    # Train the model on the entire dataset
    model.fit(X, y)
    
    # Make a prediction for the given accuracy and mia
    prediction = model.predict([[accuracy, mia]])[0]
    
    return prediction[0], prediction[1]  # Return k and theta

# Example usage
accuracy_input = 0.9
mia_input = 0.2
predicted_k, predicted_theta = train_and_predict_k_theta(df, accuracy_input, mia_input)

print(f"Predicted k: {predicted_k}")
print(f"Predicted theta: {predicted_theta}")

