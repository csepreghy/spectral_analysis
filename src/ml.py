import pandas as pd
import sys

from neural_network_classifier import run_neural_network
from xboost_classifier import run_xgboost

def create_model(df, config):
  if config['type'] == 'xgboost': model = run_xgboost(df, config)
  if config['type'] == 'neural_network': model = run_neural_network(df, config)
  
  return model


df = pd.read_pickle('../data/sdss/speclines_0-10000.pkl')

print(df.head())
print(df['OII'][5])




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
  }
}

#df = pd.read_hdf('train.h5')

model = create_model(df, configs['xgboost'])
