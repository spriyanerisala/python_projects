import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("students.csv")
print("--- Dataset ---")
print(df.head())

df['average_score'] = df[['Math', 'Science', 'English']].mean(axis=1)
df['pass_fail'] = (df['average_score'] >= 70).astype(int) 

x = df[["average_score", "Attendance"]]
y = df["pass_fail"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Performance ---")
print(f"Accuracy : {accuracy * 100:.2f}%")


cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(cm)

new_student = pd.DataFrame([[75, 85]], columns=["average_score", "Attendance"])
result = model.predict(new_student)
probability = model.predict_proba(new_student)

print(f"\n=== New Student Prediction ===")
print(f"Average Score : 75 | Attendance : 85%")
print(f"Result : {' Pass' if result[0] == 1 else '\u274c Fail'}")
print(f"Pass Probability : {probability[0][1] * 100:.2f}%")


plt.figure(figsize=(10, 6))
colors = ["red" if val == 0 else "green" for val in y]
plt.scatter(df["average_score"], df["Attendance"], c=colors, edgecolors="black", s=100)
plt.xlabel("Average Score")
plt.ylabel("Attendance %")
plt.title("Student Pass/Fail Prediction")

from matplotlib.patches import Patch
legend = [Patch(color="green", label="Pass"), Patch(color="red", label="Fail")]
plt.legend(handles=legend)

plt.tight_layout()
plt.savefig("results/logistic_regression.png", dpi=150)
plt.show()
