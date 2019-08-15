import pandas as pd
import sys
import pickle

from neural_network_classifier import run_neural_network
from gaussian_classifier import run_gaussian_clf

def create_model(df, config):
#  if config['type'] == 'xgboost': model = run_xgboost(df, config)
  if config['type'] == 'neural_network': model = run_neural_network(df, config)
  if config['type'] == 'gaussian_clf': model = run_gaussian_clf(df, config)

  return model

configs = {
  'neural_network': {
    'type': 'neural_network',
    'batch_size': 5,
    'hidden_layers': [128, 128],
    'n_epochs': 5
  },
  'xgboost': {
    'type': 'xgboost',
    'max_depth': 3,
    'learning_rate': 0.1
  },
  'gaussian_clf': {
    'type' : 'gaussian_clf',
    'kernel_val' : 1.0
  }
}

df_preprocessed = pd.read_pickle('data/preprocessed_5001-10000.pkl')

# print('df', df)
model = create_model(df_preprocessed, configs['neural_network'])
 
