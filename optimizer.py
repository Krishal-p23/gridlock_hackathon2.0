import optuna
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from tqdm import tqdm

from model import DemandModel

def objective(trial, df, params_base):

    # -----------------------
    # 1. Sample hyperparameters
    # -----------------------
    params = params_base.copy()

    params.update({
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),

        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),

        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 20.0),

        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
    })

    # -----------------------
    # 2. CV setup
    # -----------------------
    kf = KFold(n_splits=5,
               shuffle=True,
               random_state=42)

    y = df["demand"]
    x = df.drop(columns=["demand"])

    scores = []

    # -----------------------
    # 3. Cross validation loop
    # -----------------------
    for train_idx, val_idx in kf.split(x, y,):

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        model = DemandModel(params, verbose=False)

        model.fit(train_df, val_df)

        preds = model.predict(val_df)

        score = r2_score(val_df["demand"], preds)
        scores.append(score)

    return np.mean(scores)

def tune(df, params_base, n_trials=50):

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    pbar = tqdm(total=n_trials,
                desc="Optuna Tuning",
                leave=True,
                dynamic_ncols=True
                )

    def callback(study, trial):
        pbar.update(1)
        pbar.set_postfix(best=round(study.best_value, 4) if study.best_value else None)

    study.optimize(lambda trial: 
                   objective(trial, df, params_base),
                    n_trials=n_trials,
                    callbacks=[callback]
                    )
    
    pbar.close()

    print("Best R2:", study.best_value)
    print("Best params:", study.best_params)

    return study