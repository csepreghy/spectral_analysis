import pandas as pd
import sys

from neural_network_classifier import run_neural_network
# from xboost_classifier import run_xgboost
# from Caussian_classifier import run_Gauss_classi

def create_model(df, config):
#  if config['type'] == 'xgboost': model = run_xgboost(df, config)
  if config['type'] == 'neural_network': model = run_neural_network(df, config)
  if config['type'] == 'gauss_classi': model = run_Gauss_classi(df, config)

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
  'gauss_classi' :{
    'type' : 'Gauss_classi',
    'kernel_val' : 1.0
  }
}

#df = pd.read_hdf('train.h5')
df = pd.read_pickle('COMPLETE_df.pkl')
model = create_model(df, configs['neural_network'])
