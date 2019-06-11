from sklearn.gaussian_process import GaussianProcessClassifier as GPC
import pandas as pd
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import OneHotEncoder




def Gauss_classi(df):
    X = df.drop(columns={"class", "class_numbers"}).as_matrix()
    y = df["class_numbers"].get_values()

    # Make y input in the form of a hot encoder (binary matrix)
    onehot_encoder = OneHotEncoder(sparse=False)
    y_encoded = onehot_encoder.fit_transform(y.reshape(len(y), 1))

    kernel = 1.0 * RBF(1.0)

    model = GPC(kernel=kernel, random_state=0)
    model.fit(X, y)

    y_pred_train = model.predict(X)
    predictions_train = [round(value) for value in y_pred_train]
    accuracy_train = GPC.score(y, predictions_train)
    print("Accuracy on training set: %.2f%%" % (accuracy_train * 100.0))
    print(y_pred_train)

