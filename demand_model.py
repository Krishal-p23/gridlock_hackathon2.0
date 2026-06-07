from feature_pipeline import DemandFeaturePipeline


class DemandModel:
    """
    Generic training wrapper for regression models.

    Supports:
    - LightGBM
    - XGBoost
    - CatBoost
    - Any sklearn-compatible regressor

    Model-specific fit arguments are provided via fit_config.
    """

    def __init__(
        self,
        estimator_class,
        params=None,
        feature_pipeline=None,
        fit_config=None,
        verbose=False
    ):
        self.estimator_class = estimator_class
        self.params = params or {}
        self.fit_config = fit_config or {}
        self.verbose = verbose

        self.pipeline = (
            feature_pipeline
            if feature_pipeline is not None
            else DemandFeaturePipeline()
        )

        self.model = None
        self.features = None
        self.cat_cols = None

    # --------------------------------------------------
    # FEATURE PREPARATION
    # --------------------------------------------------
    def _prepare_features(self, train_df, val_df=None):

        self.pipeline.fit(train_df)

        x_train = self.pipeline.transform(train_df)

        self.features = x_train.columns.tolist()

        x_train = x_train.reindex(
            columns=self.features,
            fill_value=0
        )

        x_val = None

        if val_df is not None:
            x_val = self.pipeline.transform(val_df)

            x_val = x_val.reindex(
                columns=self.features,
                fill_value=0
            )

        self.cat_cols = (
            x_train.select_dtypes(
                include=["category"]
            )
            .columns
            .tolist()
        )

        return x_train, x_val

    # --------------------------------------------------
    # BUILD MODEL
    # --------------------------------------------------
    def _build_model(self):

        model_name = self.estimator_class.__name__

        if model_name == "XGBRegressor":

            self.model = self.estimator_class(
                **self.params,
                enable_categorical=True
            )

        elif model_name == "CatBoostRegressor":

            self.model = self.estimator_class(
                **self.params,
                verbose=self.verbose
            )

        else:

            self.model = self.estimator_class(
                **self.params
            )

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------
    def fit(self, train_df, val_df=None):

        x_train, x_val = self._prepare_features(
            train_df,
            val_df
        )

        y_train = train_df["demand"]

        self._build_model()

        fit_kwargs = self.fit_config.copy()

        if val_df is not None:

            y_val = val_df["demand"]

            model_name = self.estimator_class.__name__

            if model_name == "CatBoostRegressor":

                fit_kwargs.update({
                    "eval_set": (x_val, y_val),
                    "cat_features": self.cat_cols
                })

            else:

                fit_kwargs.update({
                    "eval_set": [(x_val, y_val)]
                })

        self.model.fit(
            x_train,
            y_train,
            **fit_kwargs
        )

        return self

    # --------------------------------------------------
    # PREDICT
    # --------------------------------------------------
    def predict(self, df):

        x = self.pipeline.transform(df)

        x = x.reindex(
            columns=self.features,
            fill_value=0
        )

        return self.model.predict(x)