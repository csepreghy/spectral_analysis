from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def run_xgboost(df, config):
  x_train = df.drop(columns={"class", "class_numbers"}).as_matrix()
  y_train = df["class_numbers"]
  print(y_train)
  #X_train = x_train.as_matrix()
  #print(np.shape(X_train))
  #print(type(X_train))

  model = XGBClassifier(max_depth=config['max_depth'], learning_rate=config['learning_rate'])
  model.fit(x_train, y_train)

  y_pred_train = model.predict(x_train)
  predictions_train = [round(value) for value in y_pred_train]
  accuracy_train = accuracy_score(y_train, predictions_train)
  print("Accuracy on training set: %.2f%%" % (accuracy_train * 100.0))
  print(y_pred_train)
