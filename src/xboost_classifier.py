from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def run_xgboost(df, config):
  x_train = df.drop(columns={"class", "class_numbers"}).as_matrix()
  y_train = df["class_numbers"].get_values()

  # Make y input in the form of a hot encoder (binary matrix)
  onehot_encoder = OneHotEncoder(sparse=False)
  y_train_encoded = onehot_encoder.fit_transform(y_train.reshape(len(y_train), 1))


  model = XGBClassifier(max_depth=config['max_depth'], learning_rate=config['learning_rate'])
  model.fit(x_train, y_train_encoded)

  y_pred_train = model.predict(x_train)
  predictions_train = [round(value) for value in y_pred_train]
  accuracy_train = accuracy_score(y_train, predictions_train)
  print("Accuracy on training set: %.2f%%" % (accuracy_train * 100.0))
  print(y_pred_train)
