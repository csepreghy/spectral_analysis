import pandas as pd

from neural_network_classifier import run_neural_network
from xboost_classifier import run_xgboost

def create_model(df, config):
  if config['type'] == 'xboost': model = run_xgboost(df, config)
  if config['type'] == 'neural_network': model = run_neural_network(df, config)
  
  return model
  


configs = {
  'neural_network': {
    'type': 'neural_network',
    'batch_size': 5,
    'hidden_layers': [128, 128],
    'n_epochs': 5
  },
  'xgboost': {
    'type': 'xgboost'
  }
}

df = pd.read_hdf('train.h5')
model = create_model(df, configs['neural_network'])
