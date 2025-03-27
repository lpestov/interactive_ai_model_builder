import pandas as pd
import pickle

# Load the CSV file
csv_path = '/path/to/your/data.csv'  # Change to your CSV path
data = pd.read_csv(csv_path)

# Load the model
model_path = '/path/to/downloaded/model.pkl'  # Change to your model path
with open(model_path, 'rb') as f:
    model = pickle.load(f)


# Make sure that there is no target
# data = data.drop('target', axis = 1)

# Make predictions
predictions = model.predict(data)

# Print the predictions
print(predictions)