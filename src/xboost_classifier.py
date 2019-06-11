from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle

def run_xgboost(df, config):
  x_train = df.drop(columns={"class"}).as_matrix()
  y_train = df["class"].get_values()

  # Make y input in the form of a hot encoder (binary matrix)
  onehot_encoder = OneHotEncoder(sparse=False)
  y_train_encoded = onehot_encoder.fit_transform(y_train.reshape(len(y_train), 1))
  print('y_train_encoded', y_train_encoded)

  model = XGBClassifier(max_depth=config['max_depth'], learning_rate=config['learning_rate'])
  model.fit(x_train, y_train_encoded)

  y_pred_train = model.predict(x_train)
  predictions_train = [round(value) for value in y_pred_train]
  accuracy_train = accuracy_score(y_train, predictions_train)
  print("Accuracy on training set: %.2f%%" % (accuracy_train * 100.0))
  print(y_pred_train)

  return model


#with open('data/sdss/FinalTable_Andrew(40-45).pkl', 'rb') as f:
#  df = pickle.load(f)

config = {
  'max_depth': 3,
  'learning_rate': 0.1
}

#print('df', df)

#run_xgboost(df, config)
