from sklearn.gaussian_process import GaussianProcessClassifier
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as le
from sklearn.metrics import accuracy_score as acc
import pickle
import pandas as pd
import numpy as np
import time as time

def run_gaussian_clf(df, config):
  df = df[0:100]
  start = time.time()
  X = df.drop(columns={"flux_list", "wavelength", "objid", "ra", "dec", "class", "spectral_lines"})
  y = df["class"]

  X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.33, random_state=42)

  kernel = 1.0 * RBF(1.0)#config['kernel_val'])
  model = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_train, y_train)

  y_pred_test = model.predict(X_test)
  accuracy_test = acc(y_test, y_pred_test)
  end = time.time()
  tt = end - start
  print("Accuracy of trained model on test set: %.2f%%" % (accuracy_test * 100.0))
  # print(y_pred_test)
  print("time :", tt)

  model.predict_proba(X_test)
  df_result_GC = pd.DataFrame(model.predict_proba(X_test))
  df_result_GC_rename = df_result_GC.rename(columns={0: "GALAXY", 1: "QSO", 2: "STAR"})
  df_result_GC_rename["predict"] = y_pred_test
  df_result_GC_rename["actual"] = y_test
