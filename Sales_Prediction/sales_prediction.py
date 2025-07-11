import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("advertising.csv")  
# 2. Preview the data
print("First 5 rows of the dataset:")
print(df.head())

# 3. Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Feature and label separation
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Make predictions
y_pred = model.predict(X_test)

# 8. Evaluate the model
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 9. Scatter Plot - Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.tight_layout()
plt.show()

# 10. Coefficients and Intercept
print("\nRegression Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
print("Intercept:", model.intercept_)

# 11. Heatmap (Correlation Matrix)
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='Blues')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()
