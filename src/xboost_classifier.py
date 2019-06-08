from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def run_xgboost(df):
  x_train = df.drop(columns={"Truth", "p_truth_E"})
  y_train = df["Truth"]

  model = XGBClassifier(max_depth=config['max_depth'], learning_rate=)
  model.fit(x_train, y_train)

  y_pred_train = model.predict(x_train)
  predictions_train = [round(value) for value in y_pred_train]
  accuracy_train = accuracy_score(y_train, predictions_train)
  print("Accuracy on training set: %.2f%%" % (accuracy_train * 100.0))
