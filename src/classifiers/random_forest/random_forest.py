from sklearn.ensemble import RandomForestClassifier
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


with open("../data/complete-splited.pkl", 'rb') as f:
    x = pickle.load(f)

df_T = pd.DataFrame(x)

df=df_T[:100]

def run_Random_Forest(df):
    start= time.time()
    X = df.drop(columns={"flux_list", "wavelength", "objid", "ra", "dec", "class", "spectral_lines"})
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, stest_size=0.33, random_state=42)


    kernel = 1.0 * RBF(1.0)#config['kernel_val'])
    model = RandomForestClassifier().fit(X_train, y_train)



    y_pred_test = model.predict(X_test)
    accuracy_test=acc(y_test, y_pred_test)
    end=time.time()
    tt= end -start
    print("Accuracy of trained model on test set: %.2f%%" % (accuracy_test * 100.0))
    # print(y_pred_test)
    print("time :", tt)

    model.predict_proba(X_test)
    df_result_rf = pd.DataFrame(model.predict_proba(X_test))
    df_result_rf_rename = df_result_rf.rename(columns={0: "GALAXY",
                                                       1: "QSO",
                                                       2: "STAR"})
    df_result_rf_rename["predict"] = y_pred_test
    df_result_rf_rename["actual"] = y_test

    df_result_rf_rename.to_pickle("../data/result_RF.pkl")


run_Random_Forest(df)