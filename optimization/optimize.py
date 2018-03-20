#!/usr/bin/env python

from hyperopt import fmin, tpe, hp, Trials
import numpy as np
import os
import sys
import signal
from optimization_data import OptimizationData
import optimization_model
import argparse
from hyopt import *
from metrics import *
import json
import tempfile
import subprocess
from kopt.utils import merge_dicts
import pymongo

def print_exp(exp_name):
    print("-" * 40 + "\nexp_name: " + exp_name)

DIR_ROOT = os.getcwd()
DIR_OUT_TRIALS = DIR_ROOT + "/trials/"
DIR_OUT_RESULTS = DIR_ROOT + "/saved_models/best/"
os.makedirs(DIR_OUT_RESULTS, exist_ok=True)

KILL_TIMEOUT = 60 * 80


class ParamValues():
    def __init__(self, lr, q, epochs, batch):
        self.batch = batch
        self.q= q
        self.lr = lr
        self.epochs = epochs

class RunFN():
    def __init__(self, metric, hyper_params, pv,
                 data_path, sep, run_on_mongodb, start_mongodb,
                 db_name, exp_name, ip, port, max_evals):
        self.metric = metric
        self.hyper_params = hyper_params
        self.values = pv
        self.path = data_path
        self.sep = sep
        self.run_on_mongodb = run_on_mongodb
        self.start_mongodb = start_mongodb
        self.db_name = db_name
        self.exp_name = exp_name
        self.ip = ip
        self.port = port
        self.max_evals = max_evals


    def __call__(self):
        m_pid=None
        w_pid=None
        if self.run_on_mongodb:
            if self.start_mongodb:
                mongodb_path = tempfile.mkdtemp()

                proc_args = ["mongod",
                             "--dbpath=%s" % mongodb_path,
                             "--noprealloc",
                             "--port="+str(self.port)]
                print("starting mongod", proc_args)
                mongodb_proc = subprocess.Popen(
                    proc_args,
                    cwd=mongodb_path,
                )
                proc_args_worker = ["hyperopt-mongo-worker",
                                    "--mongo="+str(self.ip)+":"+str(self.port)+"/"+str(self.db_name),
                                    "--poll-interval=0.1"]

                mongo_worker_proc = subprocess.Popen(
                    proc_args_worker,
                    env=merge_dicts(os.environ, {"PYTHONPATH": os.getcwd()}),
                )
                m_pid = mongodb_proc.pid
                w_pid = mongo_worker_proc.pid
            try:
                trials = CMongoTrials(self.db_name, self.exp_name,
                                      ip=self.ip, port=self.port,
                                      kill_timeout=KILL_TIMEOUT)
            except pymongo.errors.ServerSelectionTimeoutError:
                print("No mongod process detected! Please use flag --start_mongodb or"+
                      " start mongoDB and workers. Port: " + str(self.port) +
                      " Host: " + str(self.ip) + " DB name: " + str(self.db_name))
                sys.exit(0)
        else:
            trials = Trials()
        dat = OptimizationData(m_pid, w_pid, self.path, self.sep)
        if self.metric == "OutlierLoss":
            fn = CompileFN(db_name, exp_name,
                           data_fn=dat.data,
                           model_fn=optimization_model.model,
                           add_eval_metrics={"outlier_loss": OutlierLoss()},
                           loss_metric="outlier_loss",  # which metric to optimize for
                           loss_metric_mode="min",  # try to maximize the metric
                           valid_split=None,  # use 20% of the training data for the validation set
                           save_model=None,  # checkpoint the best model
                           save_results=True,  # save the results as .json (in addition to mongoDB)
                           save_dir=DIR_OUT_TRIALS)
        elif self.metric == "OutlierRecall":
            fn = CompileFN(db_name, exp_name,
                           data_fn=dat.data,
                           model_fn=optimization_model.model,
                           add_eval_metrics={"outlier_recall": OutlierRecall(theta=25, threshold=1000)},
                           loss_metric="outlier_recall",  # which metric to optimize for
                           loss_metric_mode="max",  # try to maximize the metric
                           valid_split=None,  # use 20% of the training data for the validation set
                           save_model=None,  # checkpoint the best model
                           save_results=True,  # save the results as .json (in addition to mongoDB)
                           save_dir=DIR_OUT_TRIALS)
        else:
            raise ValueError("No such metric: " + str(metric) +
                             " Available metrics for --use_metric are: 'OutlierLoss'(default), 'OutlierRecall'.")
        best = fmin(fn, hyper_params, trials=trials, algo=tpe.suggest, max_evals=self.max_evals)
        best['encoding_dim'] = self.values.q[best['encoding_dim']]
        best['batch_size'] = self.values.batch[best['batch_size']]
        best['epochs'] = self.values.epochs[best['epochs']]
        with open(DIR_OUT_RESULTS+exp_name+"_best.json", 'wt') as f:
            json.dump(best, f)
        print("----------------------------------------------------")
        print("best_parameters: " + str(best))
        print("----------------------------------------------------")
        if self.start_mongodb:
            os.killpg(os.getpgid(mongo_worker_proc.pid), signal.SIGTERM)
            os.killpg(os.getpgid(mongodb_proc.pid), signal.SIGTERM)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="run hyper-parameter optimization")
    parser.add_argument('--notest', action='store_true',
                        help="If specified runs actual optimization, if not provided runs test only.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="The path to data file with count table for which the optimization should be done,"+
                             " first row should contain column names, firs column - row names.If not provided, "+
                             "in package included sample data set (GTEX skin tissue) is used for optimization.")
    parser.add_argument('--field_sep', type=str, default=" ", help="Field separator in data file (default: space).")
    parser.add_argument('--nr_of_trials', type=int, default=1, help="Number of optimization trials (default 300).")
    parser.add_argument('--use_metric', type=str, default="OutlierLoss",
                        help="Available metrics for --use_metric are: 'OutlierLoss'(default), 'OutlierRecall'.")

    parser.add_argument('--run_on_mongodb', action='store_true',
                        help="For this option mongoDB and workers should be running with corresponding port,"+
                             " host and data base name (or flag --start_mongodb should be provided)."+
                             " If not specified runs locally.")
    parser.add_argument('--start_mongodb', action='store_true',
                        help="Start mongodb and worker from this script.")

    parser.add_argument('--db_name', type=str, default="corrector", help="Mongodb database name (default: 'corrector').")
    parser.add_argument('--exp_name', type=str, default="exp1", help="Name of experiment.")
    parser.add_argument('--ip', type=str, default="localhost", help="IP or host of mongodb (default: 'localhost').")
    parser.add_argument('--port', type=int, default=22334, help="Port of mongodb (default: 22334).")


    parser.add_argument('--optimize_only_enc_dim', action='store_true',help="Flag to optimize only encoding dimension.")
    parser.add_argument('--optimize_only_batch_size', action='store_true', help="Flag to optimize only batch size.")
    parser.add_argument('--optimize_only_epochs', action='store_true', help="Flag to optimize only number of epochs.")
    parser.add_argument('--optimize_only_lr', action='store_true', help="Flag to optimize only learning rate.")

    args = parser.parse_args()
    run_test = not args.notest
    data_path = args.data_path
    sep = args.field_sep
    metric = args.use_metric
    nr_of_trials = args.nr_of_trials
    run_on_mongodb = args.run_on_mongodb
    start_mongodb = args.start_mongodb
    db_name = args.db_name
    exp_name = args.exp_name
    ip = args.ip
    port = args.port
    only_q = args.optimize_only_enc_dim
    only_batch = args.optimize_only_batch_size
    only_epochs = args.optimize_only_epochs
    only_lr = args.optimize_only_lr

    print_exp(exp_name)


    if only_q:
        pv = ParamValues(
            lr=hp.loguniform("lr", np.log(1e-4), np.log(1e-4)),
            q=(18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
            epochs=(250,),
            batch=(32,)
        )
    elif only_batch:
         pv = ParamValues(
            lr=hp.loguniform("lr", np.log(1e-4), np.log(1e-3)),
            q=(23,),
            epochs=(250,),
            batch=(16, 32, 50, 100, 128, 200)
        )
    elif only_epochs:
        pv = ParamValues(
            lr=hp.loguniform("lr", np.log(1e-4), np.log(1e-4)),
            q=(23,),
            epochs=(100, 120, 150, 170, 200, 250, 300, 400, 500),
            batch=(32,)
        )
    elif only_lr:
        pv = ParamValues(
            lr=hp.loguniform("lr", np.log(1e-4), np.log(1e-3)),
            q=(23,),
            epochs=(250,),
            batch=(32,)
        )
    else:
        pv = ParamValues(
            lr=hp.loguniform("lr", np.log(1e-4), np.log(1e-3)),
            q = (18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
            epochs = (100, 120, 150, 170, 200, 250, 300, 400, 500),
            batch = (16, 32, 50, 100, 128, 200)
            )


    hyper_params = {
        "data": {
        },
        "model": {
            "lr": pv.lr,
            "encoding_dim": hp.choice("encoding_dim", pv.q), ##
        },
        "fit": {
            "epochs": hp.choice("epochs", pv.epochs), #
            "batch_size": hp.choice("batch_size", pv.batch)
        }
    }

    run = RunFN(metric, hyper_params, pv,
                data_path, sep, run_on_mongodb, start_mongodb,
                db_name, exp_name, ip, port, nr_of_trials)
    run()