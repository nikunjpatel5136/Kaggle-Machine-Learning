# Code you have previously used to load data
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

#set home_data as a readable table from the csv file
home_data = pd.read_csv(iowa_file_path)

# Create prediction target object and call it y
y = home_data.SalePrice

# Create X by choosing features. These are the inputted colums that will be used to make predictions for the prediction object.
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split the prediction target and features into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Specify Model using the Decision Tree Regressor
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit Model using the training data
iowa_model.fit(train_X, train_y)

# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE: {:,.0f}".format(val_mae))

from sklearn.ensemble import RandomForestRegressor

# Define the model using the Random Forest Regressor. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# Fit model using Training Data
rf_model.fit(train_X,train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(predictions, val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex5 import *
print("\nSetup complete")
