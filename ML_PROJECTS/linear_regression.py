import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os 

os.makedirs("results",exist_ok=True)
hours = [1,2,3,4,5,6,7,8,]
marks=[20,25,35,40,50,55,65,70]

x=np.array(hours).reshape(-1,1)
y=np.array(marks)

model = LinearRegression()

model.fit(x,y)
predicted_marks = model.predict([[9]])
print(f"Predicted marks for 9 hours of study : {predicted_marks}")

plt.scatter(hours,marks,color="blue")
plt.plot(x,model.predict(x),color="red")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Linear Regression: Hours Studied vs Marks")
plt.savefig("results/linear_regression_chart.png", dpi=150, bbox_inches='tight')
plt.show()
