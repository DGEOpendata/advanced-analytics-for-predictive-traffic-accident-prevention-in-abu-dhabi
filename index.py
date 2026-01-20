python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
url = 'https://data.abudhabi/transportation/traffic-accidents-2023.xlsx'
data = pd.read_excel(url)

# Preprocess the data
# Assume data has columns: 'date', 'time', 'location', 'accident_type', 'weather', 'surface_condition', 'injuries'
data['date'] = pd.to_datetime(data['date'])
data['hour'] = data['time'].apply(lambda x: int(x.split(':')[0]))
data.drop(['time'], axis=1, inplace=True)

data = pd.get_dummies(data, columns=['location', 'accident_type', 'weather', 'surface_condition'])

# Define features and target variable
y = data['injuries'] > 0  # Predict if there were injuries
X = data.drop(['injuries'], axis=1)

# Split the data
dX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
