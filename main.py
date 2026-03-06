import sklearn as sk
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])

model = sk.ensemble.RandomForestClassifier()
model.fit(X, y)

print(model.predict(np.array([[7]])))