from sklearn.svm import SVC

svm = SVC(kernel="rbf", C=3, gamma="scale")
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("SVM RBF:")
print(classification_report(y_test, y_pred_svm))
