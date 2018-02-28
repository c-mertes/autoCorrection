#!/usr/bin/env python

from hyperopt import fmin, tpe, hp, pyll
from copy import deepcopy
import numpy as np
from glob import glob
import os
import data
import model
from joblib import Parallel, delayed
import argparse
from hyopt import *
from metrics import *
import json
from autoCorrect import *
from keras.optimizers import RMSprop, Adam
from autoCorrect import autoencoder



def print_exp(exp_name):
    print("-" * 40 + "\nexp_name: " + exp_name)

DIR_ROOT = os.getcwd() #+"/optimization"
DIR_OUT_TRIALS = DIR_ROOT + "/trials/"
DIR_OUT_RESULTS = DIR_ROOT + "/saved_models/best/"

MAX_EVALS = 600
KILL_TIMEOUT = 60 * 80  # 30 minutes

DB_NAME = "corrector"
HOST = "ouga03"


# functions
# ---------
class RunFN():
    def __init__(self, exp_name, fn, hyper_params,
                 max_evals, db_name=DB_NAME):
        self.exp_name = exp_name
        self.fn = fn
        self.hyper_params = hyper_params
        self.max_evals = max_evals
        self.db_name = db_name

    def __call__(self):
        # run
        trials = CMongoTrials(self.db_name, exp_name, kill_timeout=KILL_TIMEOUT) 
        best = fmin(fn, hyper_params, trials=trials, algo=tpe.suggest, max_evals=self.max_evals)
        with open(DIR_OUT_RESULTS+exp_name+"_best.json", 'wt') as f:
            json.dump(best, f)
        print("best_parameters: " + str(best))

        
def run_trials(exp_name, fn, hyper_params,
               run_test=True,
               max_evals=MAX_EVALS):
    """run/test trials 
    """
    print_exp(exp_name)
    if run_test:
        test_fn(fn, hyper_params, save_model=None)
    else:
        run=RunFN(exp_name, fn, hyper_params, max_evals)
        run()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run hyper-parameter optimization")
    parser.add_argument('--notest', action='store_true')
    args = parser.parse_args()
    run_test = not args.notest
    
    # --------------------------------------------
    exp_name = "expX"
    print_exp(exp_name)
    # -----
    fn = CompileFN(DB_NAME, exp_name,
                   data_fn=data.data,
                   model_fn=model.model,
                   add_eval_metrics={"outlier_loss":OutlierLoss()},
                   #add_eval_metrics={"outlier_recall":OutlierRecall(theta=25, threshold=1000)},
                   loss_metric="outlier_loss", # which metric to optimize for
                   loss_metric_mode="min",  # try to maximize the metric
                   valid_split=None, # use 20% of the training data for the validation set
                   save_model=None, # checkpoint the best model
                   save_results=True, # save the results as .json (in addition to mongoDB)
                   save_dir= DIR_OUT_TRIALS)

    hyper_params = {
        "data": {
        },
        "model": {
            "lr": hp.loguniform("m_lr", np.log(1e-4), np.log(1e-3)), # 0.0001 - 0.001
            "encoding_dim": hp.choice("m_emb", (18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)), ##
        },
        "fit": {
            "epochs": hp.choice("epochs", (100, 150, 200, 250, 300, 400, 500, 600, 700, 900, 1100)), #
            "batch_size": hp.choice("batch_size", (None, 32, 50, 100, 128, 200))
        }
    }

    run_trials(exp_name, fn, hyper_params,
               run_test=run_test,
               max_evals=MAX_EVALS)
    
    # --------------------------------------------
    #exp_name = ".."
    # -----
    #c_hyper_params = deepcopy(hyper_params)
