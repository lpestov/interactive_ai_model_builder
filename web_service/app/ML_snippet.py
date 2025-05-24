import pandas as pd
import pickle

csv_path = '/path/to/your/data.csv'
data = pd.read_csv(csv_path)

model_path = '/path/to/downloaded/model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)


predictions = model.predict(data)

print(predictions)