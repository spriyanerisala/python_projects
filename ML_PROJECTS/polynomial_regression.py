import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


np.random.seed(42)

study_hours =np.round(np.random.uniform(0.5,11.5,50),1)

true_marks = -2.1 * study_hours** 2  + 28 * study_hours + 20
noise= np.random.normal(0,8,50)
exam_marks=np.clip(true_marks + noise , 5,100)
print("Student Data Sample")
print(f"{"Study Hours" : >10} | {"Exam Marks" : >10}")
print("-" * 25)
for i,j in zip(study_hours[:8],exam_marks[:8]):
    print(f"{i:>10.1f} | {j:>10.1f}")

x=study_hours.reshape(-1,1)
y=exam_marks

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(f"Training Students : {len(x_train)}")
print(f"Testing Students : {len(y_test)}")


poly = PolynomialFeatures(degree=2)
x_poly_demo = poly.fit_transform([[3],[6],[9]])

print("Polynomial Features for study hours 3,6,9")
print(f"Feature names: {poly.get_feature_names_out(['hours'])}")
print()
for hrs, feat in zip([3, 6, 9], x_poly_demo):
    print(f"hours={hrs} → features={feat}")

model = Pipeline([
    ("poly_features",PolynomialFeatures(degree=2,include_bias=True)),
    ("linear_regression",LinearRegression())
])

model.fit(x_train,y_train)

reg = model.named_steps["linear_regression"]
poly=model.named_steps["poly_features"]
names=poly.get_feature_names_out(["hours"])

print(" Model Coefficients")
print(f"Intercept : {reg.intercept_:.3f}")
for name, coef in zip(names[1:], reg.coef_[1:]):
    print(f"{name:>10} coefficient: {coef:.3f}")

print(f"\nEquation: marks = {reg.intercept_:.2f} "
      f"+ {reg.coef_[1]:.3f}×hours "
      f"+ {reg.coef_[2]:.4f}×hours²")

y_pred_train = model.predict(x_train)
y_pred_test  = model.predict(x_test)

train_r2  = r2_score(y_train, y_pred_train)
test_r2   = r2_score(y_test, y_pred_test)
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse  = mean_squared_error(y_test, y_pred_test)

print("=== Model Performance ===")
print(f"{'Metric':<15} {'Train':>10} {'Test':>10}")
print("-" * 37)
print(f"{'R² Score':<15} {train_r2:>10.4f} {test_r2:>10.4f}")
print(f"{'MSE':<15} {train_mse:>10.2f} {test_mse:>10.2f}")
print(f"{'RMSE':<15} {train_mse**0.5:>10.2f} {test_mse**0.5:>10.2f}")


print("\n=== Individual Student Predictions ===")
test_students = [[2], [4], [6], [7.5], [10]]
for hrs in test_students:
    pred = model.predict([hrs])[0]
    print(f"  {hrs[0]} hrs study → Predicted marks: {min(100,max(0,pred)):.1f}")
    
    

hours_range = np.linspace(0, 12, 1000).reshape(-1, 1)
predicted_marks = model.predict(hours_range)

optimal_hours = hours_range[np.argmax(predicted_marks)][0]
max_marks = predicted_marks.max()

print(f"Optimal study hours : {optimal_hours:.1f} hrs")
print(f"Maximum marks       : {max_marks:.1f} / 100")
print(f"Tip: If your reading time is {optimal_hours:.1f} hours, you can achieve up to {max_marks:.0f} marks!")    

degrees = [1, 2, 3, 5, 8]
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

for ax, deg in zip(axes, degrees):
    m = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('reg',  LinearRegression())
    ]).fit(x_train, y_train)

    test_r2 = r2_score(y_test, m.predict(x_test))

    x_plot = np.linspace(0, 12, 300).reshape(-1, 1)
    y_plot = np.clip(m.predict(x_plot), 0, 100)

    ax.scatter(x_train, y_train, s=15, c='steelblue', alpha=0.7, label='Train')
    ax.scatter(x_test,  y_test,  s=15, c='tomato',    alpha=0.9, label='Test')
    ax.plot(x_plot, y_plot, color='green', lw=2)
    ax.set_title(f'Degree {deg}\nR²={test_r2:.3f}', fontsize=10)
    ax.set_xlabel('Study hours')
    ax.set_ylabel('Marks')
    ax.set_ylim(0, 105)
    if deg == 1:
        ax.set_title(f'Degree {deg} — UNDERFIT\nR²={test_r2:.3f}', fontsize=9)
    elif deg == 2:
        ax.set_title(f'Degree {deg} — BEST FIT\nR²={test_r2:.3f}', fontsize=9)
    elif deg >= 6:
        ax.set_title(f'Degree {deg} — OVERFIT\nR²={test_r2:.3f}', fontsize=9)

plt.suptitle('Student marks: Underfitting → Best fit → Overfitting', fontsize=12)
plt.tight_layout()
plt.savefig("results/polynomial_regression.png", dpi=150)
plt.show()