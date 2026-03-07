from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

y_pred = model.predict(X_val)
cm = confusion_matrix(y_val, y_pred, labels=model.classes_)
ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(cmap='Blues'); plt.show()

fi = model.get_feature_importance(prettified=True)
print(fi.head(30))
