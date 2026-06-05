import lightgbm as lgb

from feature_pipeline import DemandFeaturePipeline


class DemandModel:

    def __init__(self, params, verbose=False):
        self.pipeline = DemandFeaturePipeline()
        self.model = None
        self.params = params
        self.cat_cols = None   # will be inferred after transform
        self.verbose = verbose

    # -----------------------
    # FIT
    # -----------------------
    def fit(self, train_df, val_df):

        self.pipeline.fit(train_df)

        x_train = self.pipeline.transform(train_df)
        x_val = self.pipeline.transform(val_df)

        x_train = x_train.reindex(columns=self.pipeline.features, fill_value=0)
        x_val = x_val.reindex(columns=self.pipeline.features, fill_value=0)

        y_train = train_df["demand"]
        y_val = val_df["demand"]

        self.cat_cols = x_train.select_dtypes(include=["category"]).columns.tolist()
        for col in self.cat_cols:
            x_train[col] = x_train[col].astype("category")
            x_val[col] = x_val[col].astype("category")

        train_data = lgb.Dataset(
            x_train,
            label=y_train,
            categorical_feature=self.cat_cols
        )

        val_data = lgb.Dataset(
            x_val,
            label=y_val,
            categorical_feature=self.cat_cols
        )

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=5000,
            callbacks=[lgb.early_stopping(100, verbose=self.verbose),
                       lgb.log_evaluation(100, show_stdv=False) if self.verbose else None]
        )

    # -----------------------
    # PREDICT
    # -----------------------
    def predict(self, df):

        x = self.pipeline.transform(df)

        # safety: enforce same columns as training
        if hasattr(self.pipeline, "features") and self.pipeline.features is not None:
            x = x.reindex(columns=self.pipeline.features, fill_value=0)

        return self.model.predict(x)