from .correctors import *
from .data_utils import DataLoaderWithPred, DataLoader, DataReader
# kopt and hyoperot imports
from kopt import CompileFN, KMongoTrials, test_fn
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#class LossOptimization():
#    def __init__(self):
#        pass

    def data(inject_outliers=False,inject_zeros=True):
        dr=data_utils.DataReader()
        counts = dr.read_gtex_skin()
        cook = data_utils.DataCooker(counts)
        data = cook.data(inject_outliers=inject_outliers,inject_zeros=inject_zeros)
        return data

    def model(train_data, lr=0.001,
          encoding_dim=128):
        size = train_data[0]["inp"].shape[1]
        ae = autoencoder.Autoencoder(choose_autoencoder=True, size=size, 
                         encoding_dim=encoding_dim)
        ae.model.compile(optimizer=Adam(lr=lr), loss=ae.loss)
        model = autoenc.model
        return model

#    def objective():
        db_name="correct"
        exp_name="myexp1"
        objective = CompileFN(db_name, exp_name,
                              data_fn=data,
                              model_fn=model,
                              loss_metric="loss", # which metric to optimize for
                              loss_metric_mode="min",  # try to maximize the metric
                              valid_split=0.2, # use 20% of the training data for the validation set
                              save_model='best', # checkpoint the best model
                              save_results=True, # save the results as .json (in addition to mongoDB)
                              save_dir="./saved_models/")  # p
#        return objective

    hyper_params = {
        "data": {
            "inject_outliers": hp.choice('inj_out', (True, False)),
            "inject_zeros": hp.choice('inj_z', (True, False)),
        },
        "model": {
            "lr": hp.loguniform("m_lr", np.log(1e-4), np.log(1e-2)), # 0.0001 - 0.01
            "encoding_dim": hp.choice("m_emb", (2, 4, 8, 10, 14, 18, 24)),
        },
        "fit": {
            "epochs": hp.choice("epochs", (80, 100, 120, 200, 400))
        }
    }

test_fn(objective, hyper_params)

#trials = Trials()
#best = fmin(objective, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)
#trials = KMongoTrials(db_name, exp_name,
#                      ip="localhost",
#	                  port=22334)
#trials = Trials()
#best = fmin(objective, hyper_params, trials=trials, algo=tpe.suggest, max_evals=2)