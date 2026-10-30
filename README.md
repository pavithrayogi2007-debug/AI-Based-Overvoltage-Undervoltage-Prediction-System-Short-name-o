# AI-Based-Overvoltage-Undervoltage-Prediction-System-Short-name-o
AI-based system using Python and Machine Learning to detect and predict overvoltage and undervoltage conditions.

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Create Dataset
np.random.seed(42)

data = pd.DataFrame({
    'Voltage': np.random.uniform(150, 280, 500),
    'Current': np.random.uniform(1, 20, 500),
    'Frequency': np.random.uniform(48, 52, 500),
    'Power_Factor': np.random.uniform(0.5, 1.0, 500)
})

# Step 2: Define Voltage Conditions
def voltage_status(voltage):
    if voltage < 207:
        return 'Undervoltage'
    elif voltage > 253:
        return 'Overvoltage'
    else:
        return 'Normal'

data['Status'] = data['Voltage'].apply(voltage_status)

# Step 3: Split Features and Target
X = data[['Voltage', 'Current',
          'Frequency', 'Power_Factor']]

y = data['Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train Machine Learning Model
model = RandomForestClassifier(n_estimators=100,
                               random_state=42)

model.fit(X_train, y_train)

# Step 5: Model Evaluation
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy * 100, "%")
print(classification_report(y_test, predictions))

# Step 6: Predict Voltage Condition
print("\n--- Voltage Prediction System ---")

voltage = float(input("Enter Voltage: "))
current = float(input("Enter Current: "))
frequency = float(input("Enter Frequency: "))
pf = float(input("Enter Power Factor: "))

new_data = pd.DataFrame(
    [[voltage, current, frequency, pf]],
    columns=['Voltage', 'Current',
             'Frequency', 'Power_Factor']
)

result = model.predict(new_data)

print("\nPredicted Voltage Condition:", result[0])

if result[0] == 'Normal':
    print("Voltage is Normal")
elif result[0] == 'Overvoltage':
    print("Warning! Overvoltage Detected")
else:
    print("Warning! Undervoltage Detected")
    
