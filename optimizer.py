import optuna
import numpy as np

from tqdm import tqdm

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

from demand_model import DemandModel


# --------------------------------------------------
# PARAMETER SAMPLER
# --------------------------------------------------
def sample_parameters(
    trial,
    params_base,
    search_space
):
    """
    Generic Optuna parameter sampler.

    search_space format:

    {
        "learning_rate": {
            "type": "float",
            "low": 0.005,
            "high": 0.05,
            "log": True
        },

        "num_leaves": {
            "type": "int",
            "low": 16,
            "high": 256
        }
    }
    """

    params = params_base.copy()

    for param_name, cfg in search_space.items():

        param_type = cfg["type"]

        if param_type == "float":

            params[param_name] = trial.suggest_float(
                param_name,
                cfg["low"],
                cfg["high"],
                log=cfg.get("log", False)
            )

        elif param_type == "int":

            params[param_name] = trial.suggest_int(
                param_name,
                cfg["low"],
                cfg["high"]
            )

        elif param_type == "categorical":

            params[param_name] = trial.suggest_categorical(
                param_name,
                cfg["choices"]
            )

        else:

            raise ValueError(
                f"Unsupported parameter type: {param_type}"
            )

    return params


# --------------------------------------------------
# OBJECTIVE
# --------------------------------------------------
def objective(
    trial,
    df,
    estimator_class,
    params_base,
    search_space,
    fit_config,
    scoring_fn,
    cv,
    feature_pipeline=None,
    verbose=False
):

    params = sample_parameters(
        trial,
        params_base,
        search_space
    )

    scores = []

    y = df["demand"]
    x = df.drop(columns=["demand"])

    for fold_idx, (train_idx, val_idx) in enumerate(
        cv.split(x, y)
    ):

        train_df = df.iloc[train_idx].copy()

        val_df = df.iloc[val_idx].copy()

        model = DemandModel(
            estimator_class=estimator_class,
            params=params,
            fit_config=fit_config,
            feature_pipeline=feature_pipeline,
            verbose=verbose
        )

        model.fit(
            train_df,
            val_df
        )

        preds = model.predict(
            val_df
        )

        score = scoring_fn(
            val_df["demand"],
            preds
        )

        scores.append(score)

        trial.report(
            np.mean(scores),
            fold_idx
        )

        if trial.should_prune():

            raise optuna.TrialPruned()

    return float(np.mean(scores))


# --------------------------------------------------
# TUNE
# --------------------------------------------------
def tune(
    df,
    estimator_class,
    search_space,
    params_base=None,
    fit_config=None,
    feature_pipeline=None,
    scoring_fn=r2_score,
    n_trials=50,
    n_splits=5,
    random_state=42,
    direction="maximize",
    verbose=False
):
    """
    Generic Optuna tuner.

    Parameters
    ----------
    df : DataFrame

    estimator_class :
        LGBMRegressor,
        XGBRegressor,
        CatBoostRegressor,
        RandomForestRegressor,
        etc.

    search_space : dict
        Hyperparameter search space

    params_base : dict
        Fixed parameters

    fit_config : dict
        Extra fit() arguments

    scoring_fn : callable
        r2_score,
        mean_absolute_error,
        etc.
    """

    params_base = params_base or {}
    fit_config = fit_config or {}

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    study = optuna.create_study(
        direction=direction,
        pruner=optuna.pruners.MedianPruner(
            n_warmup_steps=2
        )
    )

    pbar = tqdm(
        total=n_trials,
        desc="Optuna Tuning",
        leave=True,
        dynamic_ncols=True
    )

    def callback(
        study,
        trial
    ):
        pbar.update(1)

        if len(study.trials) > 0:

            pbar.set_postfix(
                best=round(
                    study.best_value,
                    5
                )
            )

    study.optimize(
        lambda trial: objective(
            trial=trial,
            df=df,
            estimator_class=estimator_class,
            params_base=params_base,
            search_space=search_space,
            fit_config=fit_config,
            scoring_fn=scoring_fn,
            cv=cv,
            feature_pipeline=feature_pipeline,
            verbose=verbose
        ),
        n_trials=n_trials,
        callbacks=[callback]
    )

    pbar.close()

    print(
        "\nBest Score:",
        study.best_value
    )

    print(
        "\nBest Parameters:"
    )

    for k, v in study.best_params.items():
        print(f"{k}: {v}")

    return study