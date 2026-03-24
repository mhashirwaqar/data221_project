import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, consufion_matrix, ConfusionMatrixDisplay, classification_report

df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df1.dataframeName = 'creditcard.csv'




X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.2, random_state = 42, stratify = y)
#train decision tree
dt_model = DecisionTreeClassifier(random_state = 42, max_depth = 4)
dt_model.fit (X_train, y_train)
#predictions
y_pred = dt_model.predict (X_test)
y_prob = dt_model.predict_proba (X_test)[:1]

#Evaluation metrics 
acc = accuracy_score (y_test, y_pred)
prec = precision_score (y_test, y_pred, zero_divison = 0)
rec = recall_score (y_test, y_pred, zero_divison = 0)
f1 = f1_score (y_test, y_pred, zero_divison = 0)
roc_auc = roc_auc_score (y_test, y_prob)

print ("Decision Tree Results")
print ("Accuracy:", round(acc, 4))
print ("Precision:", round(prec, 4))
print ("Recall:", round(rec, 4))
print ("F1-score:", round(f1, 4))
print ("ROC-AUC:", round(roc_auc , 4))

print ("\nClassification Report:")
print (classification_report(y_test, y_pred, zero_division = 0)
#Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

disp.plot()
ptl.title("Decision Tree Confusion Matrix")
ptl.savefig("decison_tree_confusion_matrix.png", bbox_inches = "tight")
ptl.show()

#Feature Importance Visualization 
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": dt_model.feature_importances})
feature_importance = feature_importance.sort_values(by = "Importance", ascending = False)

top_features = feature_importance.head(10)


