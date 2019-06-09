from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def roccurves(y_true, y_prob):
    x, y, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(x, y)
    return x, y, roc_auc

#plot roc curve
#plt.plot(x, y, lw=lw, label='(area = %0.2f)' % roc_auc)