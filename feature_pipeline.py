import pandas as pd
import numpy as np

class DemandFeaturePipeline:

    def __init__(self):
        self.temp_median = None
        self.global_mean = None
        self.geohash_te_map = None

        self.road_map = {
            'Highway': 3,
            'Street': 2,
            'Residential': 1
        }

        self.cat_cols = [
            "RoadType",
            "Weather",
            "temp_bin",
            "demand_period",
            "time_bucket",
            "weather_temp"
        ]

        self.features = None

    # -----------------------
    # FIT
    # -----------------------
    def fit(self, df):

        df = df.copy()

        self.temp_median = df["Temperature"].median()
        self.global_mean = df["demand"].mean()

        # smoothed geohash encoding
        stats = df.groupby("geohash")["demand"].agg(["mean", "count"])
        alpha = 20

        self.geohash_te_map = (
            (stats["mean"] * stats["count"] + self.global_mean * alpha)
            / (stats["count"] + alpha)
        ).to_dict()

        return self

    # -----------------------
    # TRANSFORM
    # -----------------------
    def transform(self, df):

        df = df.copy()

        # -----------------------
        # 1. Clean categorical
        # -----------------------
        cat_cols_raw = [
            'geohash','RoadType','LargeVehicles',
            'Landmarks','Weather'
        ]

        for c in cat_cols_raw:
            df[c] = df[c].astype(str).str.strip()

        # -----------------------
        # 2. Missing flags
        # -----------------------
        df['temp_missing'] = df['Temperature'].isna().astype(int)
        df['weather_missing'] = df['Weather'].isna().astype(int)
        df['roadtype_missing'] = df['RoadType'].isna().astype(int)

        # -----------------------
        # 3. Fill temperature
        # -----------------------
        df['Temperature'] = df['Temperature'].fillna(self.temp_median)

        # -----------------------
        # 4. Time features
        # -----------------------
        df[['hour','minute']] = df['timestamp'].str.split(':', expand=True).astype(int)

        df['total_minutes'] = df['hour'] * 60 + df['minute']

        df['sin_time'] = np.sin(2*np.pi*df['total_minutes']/1440)
        df['cos_time'] = np.cos(2*np.pi*df['total_minutes']/1440)

        # -----------------------
        # 5. Time buckets
        # -----------------------
        df['time_bucket'] = df['hour'].apply(
            lambda h: 'Night' if h < 6 else
                      'Morning' if h < 12 else
                      'Afternoon' if h < 18 else
                      'Evening'
        )

        df['demand_period'] = df['hour'].apply(
            lambda h: 'Night' if h <= 4 else
                      'MorningRise' if h <= 9 else
                      'Peak' if h <= 14 else
                      'Decline' if h <= 17 else
                      'LowDemand' if h <= 20 else
                      'Recovery'
        )

        # -----------------------
        # 6. Temperature features
        # -----------------------
        df['temp_bin'] = pd.cut(
            df['Temperature'],
            bins=[-np.inf,10,20,30,40,np.inf],
            labels=['VeryCold','Cold','Moderate','Warm','Hot']
        )

        df['temp_sq'] = df['Temperature'] ** 2

        df['weather_temp'] = (
            df['Weather'].astype(str) + "_" + df['temp_bin'].astype(str)
        )

        # -----------------------
        # 7. Binary encoding
        # -----------------------
        df['Landmarks_bin'] = df['Landmarks'].map({'Yes':1,'No':0}).fillna(0)
        df['LargeVehicles_bin'] = df['LargeVehicles'].map({'Allowed':1,'Not Allowed':0}).fillna(0)

        # -----------------------
        # 8. Road features
        # -----------------------
        df['road_score'] = df['RoadType'].map(self.road_map).fillna(1)
        df['road_capacity'] = df['road_score'] * df['NumberofLanes']
        df['lane_vehicle_interaction'] = df['NumberofLanes'] * df['LargeVehicles_bin']

        # -----------------------
        # 9. Cyclic day
        # -----------------------
        df['sin_day'] = np.sin(2*np.pi*df['day']/7)
        df['cos_day'] = np.cos(2*np.pi*df['day']/7)

        # -----------------------
        # 10. Geohash encoding
        # -----------------------
        df['geohash_te'] = df['geohash'].map(self.geohash_te_map)
        df['geohash_te'] = df['geohash_te'].fillna(self.global_mean)

        # -----------------------
        # 11. SAFE categorical casting
        # -----------------------
        for c in self.cat_cols:
            df[c] = df[c].astype('category')

        # -----------------------
        # 12. FEATURE LOCK (VERY IMPORTANT)
        # -----------------------
        if self.features is not None:
            df = df.reindex(columns=self.features, fill_value=0)

        drop_cols = [
        "geohash",
        "timestamp",
        "LargeVehicles",
        "Landmarks"
        ]

        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        for col in self.cat_cols:
            if col in df.columns and df[col].dtype == 'str':
                df[col] = df[col].astype('category')

        feature_cols = [
            "sin_time",
            "cos_time",
            "sin_day",
            "cos_day",
            "hour",

            "Temperature",
            "temp_sq",

            "NumberofLanes",
            "road_capacity",
            "road_score",

            "Landmarks_bin",
            "LargeVehicles_bin",
            "lane_vehicle_interaction",

            "RoadType",
            "Weather",
            "temp_bin",
            "demand_period",
            "time_bucket",
            "weather_temp",

            "geohash_te"
        ]

        df = df[feature_cols]

        return df
