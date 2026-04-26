import numpy as np
from sklearn.linear_model import LinearRegression
import pickle
x = np.array([
    [2, 60],
    [4, 70],
    [6, 80],
    [8, 90],
    [10, 95]
])

y = np.array([50, 60, 70, 85, 95])

model=LinearRegression()
model.fit(x,y)

pickle.dump(model,open('model.pkl','wb'))