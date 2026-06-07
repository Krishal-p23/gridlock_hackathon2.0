import pandas as pd
import numpy as np


class DemandFeaturePipeline:

    def __init__(self):

        self.temp_median = None
        self.global_mean = None
        self.geohash_te_map = None

        self.road_map = {
            "Highway": 3,
            "Street": 2,
            "Residential": 1
        }

        self.cat_cols = [
            "RoadType",
            "Weather",
            "temp_bin",
            "demand_period",
            "time_bucket",
            "weather_temp"
        ]

        self.features = []

    # --------------------------------------------------
    # FIT
    # --------------------------------------------------
    def fit(self, df):

        df = df.copy()

        self.temp_median = df["Temperature"].median()

        self.global_mean = df["demand"].mean()

        stats = (
            df.groupby("geohash")["demand"]
            .agg(["mean", "count"])
        )

        alpha = 20

        self.geohash_te_map = (
            (
                stats["mean"] * stats["count"]
                + self.global_mean * alpha
            )
            / (stats["count"] + alpha)
        ).to_dict()

        return self

    # --------------------------------------------------
    # CLEAN CATEGORICALS
    # --------------------------------------------------
    def _clean_categories(self, df):

        cat_cols_raw = [
            "geohash",
            "RoadType",
            "LargeVehicles",
            "Landmarks",
            "Weather"
        ]

        for col in cat_cols_raw:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
            )

        return df

    # --------------------------------------------------
    # TIME FEATURES
    # --------------------------------------------------
    def _add_time_features(self, df):

        df[["hour", "minute"]] = (
            df["timestamp"]
            .str.split(":", expand=True)
            .astype(int)
        )

        df["total_minutes"] = (
            df["hour"] * 60
            + df["minute"]
        )

        df["sin_time"] = np.sin(
            2 * np.pi * df["total_minutes"] / 1440
        )

        df["cos_time"] = np.cos(
            2 * np.pi * df["total_minutes"] / 1440
        )

        df["time_bucket"] = df["hour"].apply(
            lambda h:
            "Night" if h < 6 else
            "Morning" if h < 12 else
            "Afternoon" if h < 18 else
            "Evening"
        )

        df["demand_period"] = df["hour"].apply(
            lambda h:
            "Night" if h <= 4 else
            "MorningRise" if h <= 9 else
            "Peak" if h <= 14 else
            "Decline" if h <= 17 else
            "LowDemand" if h <= 20 else
            "Recovery"
        )

        self.features.extend([
            "hour",
            "sin_time",
            "cos_time",
            "time_bucket",
            "demand_period"
        ])

        return df

    # --------------------------------------------------
    # TEMPERATURE FEATURES
    # --------------------------------------------------
    def _add_temperature_features(self, df):

        df["Temperature"] = (
            df["Temperature"]
            .fillna(self.temp_median)
        )

        df["temp_bin"] = pd.cut(
            df["Temperature"],
            bins=[-np.inf, 10, 20, 30, 40, np.inf],
            labels=[
                "VeryCold",
                "Cold",
                "Moderate",
                "Warm",
                "Hot"
            ]
        )

        df["temp_sq"] = (
            df["Temperature"] ** 2
        )

        df["weather_temp"] = (
            df["Weather"].astype(str)
            + "_"
            + df["temp_bin"].astype(str)
        )

        self.features.extend([
            "Temperature",
            "temp_sq",
            "temp_bin",
            "weather_temp"
        ])

        return df

    # --------------------------------------------------
    # ROAD FEATURES
    # --------------------------------------------------
    def _add_road_features(self, df):

        df["Landmarks_bin"] = (
            df["Landmarks"]
            .map({
                "Yes": 1,
                "No": 0
            })
            .fillna(0)
        )

        df["LargeVehicles_bin"] = (
            df["LargeVehicles"]
            .map({
                "Allowed": 1,
                "Not Allowed": 0
            })
            .fillna(0)
        )

        df["road_score"] = (
            df["RoadType"]
            .map(self.road_map)
            .fillna(1)
        )

        df["road_capacity"] = (
            df["road_score"]
            * df["NumberofLanes"]
        )

        df["lane_vehicle_interaction"] = (
            df["NumberofLanes"]
            * df["LargeVehicles_bin"]
        )

        self.features.extend([
            "NumberofLanes",
            "road_score",
            "road_capacity",
            "Landmarks_bin",
            "LargeVehicles_bin",
            "lane_vehicle_interaction",
            "RoadType"
        ])

        return df

    # --------------------------------------------------
    # DAY FEATURES
    # --------------------------------------------------
    def _add_day_features(self, df):

        df["sin_day"] = np.sin(
            2 * np.pi * df["day"] / 7
        )

        df["cos_day"] = np.cos(
            2 * np.pi * df["day"] / 7
        )

        self.features.extend([
            "sin_day",
            "cos_day"
        ])

        return df

    # --------------------------------------------------
    # GEOHASH TARGET ENCODING
    # --------------------------------------------------
    def _add_geohash_features(self, df):

        df["geohash_te"] = (
            df["geohash"]
            .map(self.geohash_te_map)
            .fillna(self.global_mean)
        )

        self.features.append(
            "geohash_te"
        )

        return df

    # --------------------------------------------------
    # CATEGORICAL CASTING
    # --------------------------------------------------
    def _cast_categories(self, df):

        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].astype(
                    "category"
                )

        self.features.append(
            "Weather"
        )

        return df

    # --------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------
    def transform(self, df):

        df = df.copy()

        # reset feature registry
        self.features = []

        df = self._clean_categories(df)

        df = self._add_time_features(df)

        df = self._add_temperature_features(df)

        df = self._add_road_features(df)

        df = self._add_day_features(df)

        df = self._add_geohash_features(df)

        df = self._cast_categories(df)

        drop_cols = [
            "geohash",
            "timestamp",
            "LargeVehicles",
            "Landmarks"
        ]

        df = df.drop(
            columns=[
                c for c in drop_cols
                if c in df.columns
            ]
        )

        return df[self.features]