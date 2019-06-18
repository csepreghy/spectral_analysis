from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pickle
import pandas as pd


def run_xgboost(df, config):
  x_train = df.drop(columns={"class"}).as_matrix()
  y_train = df["class"].get_values()
  print('y_train', y_train)

  # Make y input in the form of a hot encoder (binary matrix)
  onehotencoder = OneHotEncoder(categorical_features = [0])
  x = onehotencoder.fit_transform(y_train).toarray()
  print('x', x)
  # y_train_encoded = onehot_encoder.fit_transform(y_train.reshape(len(y_train), 1))
  print('y_train_encoded', y_train_encoded)

  data['class'] = pd.Categorical(data['class'])
  dataDummies = pd.get_dummies(data['class'], prefix='category')
  data = pd.concat([data, dataDummies], axis=1)

  model = XGBClassifier(max_depth=config['max_depth'], learning_rate=config['learning_rate'])
  model.fit(x_train, y_train_encoded)

  y_pred_train = model.predict(x_train)
  predictions_train = [round(value) for value in y_pred_train]
  accuracy_train = accuracy_score(y_train, predictions_train)
  print("Accuracy on training set: %.2f%%" % (accuracy_train * 100.0))
  print(y_pred_train)

  return model


df = pd.read_pickle('COMPLETE_df.pkl')

config = {
  'max_depth': 3,
  'learning_rate': 0.1
}
 
run_xgboost(df, config)
