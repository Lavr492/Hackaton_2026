from catboost import CatBoostClassifier
from sklearn.metrics import classification_report

model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function="MultiClass",
    verbose=False
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred = y_pred.flatten()

print("CatBoost:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
