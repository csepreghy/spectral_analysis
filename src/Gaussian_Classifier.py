from sklearn.gaussian_process import GaussianProcessClassifier
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder as le
import pickle
import pandas as pd
import numpy as np
import time as time


with open("../data/complete-splited.pkl", 'rb') as f:
    x = pickle.load(f)
df = pd.DataFrame(x)
# configs={
#     'gauss_classi': {
#         'type': 'Gauss_classi',
#         'kernel_val': 1.0
#     }

# }

def run_Gauss_classi(df):
    start= time.time()
    X = df.drop(columns={"flux_list", "wavelength", "objid", "ra", "dec", "class", "spectral_lines"})
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.33, random_state=42)

    # Make y input in the form of a hot encoder (binary matrix)
    #don't need that

    kernel = 1.0 * RBF(1.0)#config['kernel_val'])
    model = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(X_train, y_train)



    y_pred_test = model.predict(X_test)
    predictions_test = [round(value) for value in y_pred_test]
    accuracy_train = model.score(y, predictions_test)
    end=time.time()
    tt= end -start
    print("Accuracy on training set: %.2f%%" % (accuracy_train * 100.0))
    print(y_pred_test)
    print("time :", tt)


run_Gauss_classi(df)