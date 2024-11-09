import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # You could also try LinearRegression or another model
from sklearn.metrics import mean_squared_error

# Input data
k = [0.5076, 6.6256, 8.8894, 5.6432, 7.5653, 9.6156, 9.103, 9.4447, 9.6583, 9.5729, 9.402, 9.1884, 8.9322, 9.6156, 9.1884, 9.9146, 9.3593, 9.9573, 9.2739, 9.7864, 9.1884, 9.6583, 8.8894, 9.3166, 9.7864, 8.9749, 9.3166, 9.701, 8.8894, 9.2312, 9.5302, 9.8719, 9.0176, 9.3166, 9.5729, 9.8719, 8.9749, 9.2312, 9.4874, 9.7437, 10.0, 9.0603, 9.3166, 9.5302, 9.7437, 9.9573]
theta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]
mia = [0.08434114843375096, 0.08231704248921375, 0.062880898039201, 0.08810244822154573, 0.0769610411361284, 0.08369755902156667, 0.07436508195530674, 0.0822249402556695, 0.07166928669035968, 0.06849941642469226, 0.07027205298533169, 0.07176236622689079, 0.07991962317587344, 0.07705362944229147, 0.08010424424166136, 0.0829615207143473, 0.07352923328811553, 0.07807153542015544, 0.07334339422420959, 0.07455073126429625, 0.07529298403543135, 0.07408654335269817, 0.06859279146561167, 0.07436508195530674, 0.06812582904760044, 0.07631268284916351, 0.07529298403543135, 0.07547846117590556, 0.07510747248680548, 0.0822249402556695, 0.07455073126429625, 0.07677583880204963, 0.0764053312024053, 0.0693394780786129, 0.0701788346212465, 0.07760897932698288, 0.07436508195530674, 0.07705362944229147, 0.0820407103362094, 0.08010424424166136, 0.06896620446472282, 0.08194858264716733, 0.08038111194681256, 0.06961934188005707, 0.08837711061741292, 0.08148781684626792]

# Prepare the features and target
X = np.column_stack((k, theta))  # Combine k and theta into a feature matrix
y = np.array(mia)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestRegressor(random_state=42)  # Replace with LinearRegression() if you want linear regression

# Train the model
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Function to predict mia
def predict_mia(k_value, theta_value):
    input_data = np.array([[k_value, theta_value]])
    return model.predict(input_data)[0]

# Example usage
predicted_mia = predict_mia(9.0, 0.0001)
print(f"Predicted mia for k=9.0 and theta=0.0001: {predicted_mia}")
