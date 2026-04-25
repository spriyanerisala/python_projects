import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


df = pd.read_csv("house_data.csv")
print("--- Dataset ---")
print(df.head())
print(f"\nTotal Rows : {len(df)}")

x = df[["area_sqft", "bedrooms", "age_years"]]
y = df["price_lakhs"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"\n--- Model Performance ---")
print(f"MAE      : {mae:.2f} Lakhs")
print(f"R2 Score : {r2:.2f}")

new_house = pd.DataFrame([[1500, 3, 10]], columns=["area_sqft", "bedrooms", "age_years"])
predicted_price = model.predict(new_house)
print(f"\n=== New House Prediction ===")
print(f"Area: 1500 sqft | Bedrooms: 3 | Age: 10 yrs")
print(f"Predicted Price : ₹ {predicted_price[0]:.2f} Lakhs")


area_range = np.linspace(df["area_sqft"].min(), df["area_sqft"].max(), 100).reshape(-1, 1)
avg_bedrooms = df["bedrooms"].mean()
avg_age = df["age_years"].mean()

predict_input = pd.DataFrame(
    np.column_stack([area_range, np.full(len(area_range), avg_bedrooms), np.full(len(area_range), avg_age)]),
    columns=["area_sqft", "bedrooms", "age_years"]
)
predicted_line = model.predict(predict_input)


plt.figure(figsize=(10, 6))
plt.scatter(df["area_sqft"], df["price_lakhs"], color="blue", edgecolors="black", s=80, label="Data Points")
plt.plot(area_range, predicted_line, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Area (sqft)")
plt.ylabel("Price (Lakhs)")
plt.title("House Price Prediction : Area vs Price")
plt.legend()
plt.tight_layout()
plt.savefig("results/house_price_prediction.png", dpi=150)
plt.show()
