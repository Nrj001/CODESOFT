import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')

# Select useful features
features = ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3', 'Duration', 'Year', 'Rating']
df = df[features]

# Drop missing values
df.dropna(inplace=True)

# Clean Duration column
df['Duration'] = df['Duration'].str.replace(' min', '').astype(int)

# Clean Year column: remove parentheses and convert to int
df['Year'] = df['Year'].str.replace('[()]', '', regex=True).astype(int)

# Encode categorical variables
le = LabelEncoder()
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop('Rating', axis=1)
y = df['Rating']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R-squared Score (R2): {r2:.3f}")

# Sample predictions vs actual
print("\nSample predictions:")
for actual, pred in list(zip(y_test[:5], y_pred[:5])):
    print(f"Actual: {actual}, Predicted: {pred:.2f}")
