
# Demand Forecasting System - LightGBM Pipeline Documentation

## 1. Overview

This project implements a **spatio-temporal demand forecasting system** using LightGBM regression.  
The goal is to predict continuous demand values (0–1 range) using:

- Spatial information (geohash → location proxy)
- Temporal patterns (hour, day, cyclic encodings)
- Weather and temperature conditions
- Road infrastructure characteristics
- Landmark and vehicle constraints

The final model achieves **high predictive accuracy (R² ≈ 0.97+)**, indicating strong learning of underlying demand patterns.

---

## 2. Problem Formulation

We model demand as:

\[
Demand = f(Location, Time, Weather, Temperature, Infrastructure)
\]

Where:
- Location → geohash (categorical spatial ID)
- Time → cyclic temporal behavior
- Weather → environmental condition
- Infrastructure → road capacity, lanes, restrictions
- Interaction effects → nonlinear combinations

---

## 3. Exploratory Data Analysis (EDA) Insights

### 3.1 Temporal Patterns
- Demand follows a **non-linear daily cycle**
- Peak demand observed around mid-day (not traditional commute peaks)
- Cyclic encoding performs better than raw hour features

### 3.2 Spatial Patterns
- Strong variation across geohashes
- Some locations consistently high demand, others low
- Geohash acts as a dominant predictive feature

### 3.3 Weather Effects
- Weather impacts demand non-linearly
- Temperature modifies weather influence
- Rain/Fog conditions reduce mobility differently per region

### 3.4 Infrastructure Effects
- More lanes → higher demand capacity
- Landmarks significantly increase demand
- Vehicle restrictions reduce accessibility and demand

---

## 4. Feature Engineering

### 4.1 Time Features
- hour, minute extracted from timestamp
- total_minutes computed
- Cyclic encoding:
  - sin_time = sin(2π * time / 1440)
  - cos_time = cos(2π * time / 1440)

- Day cyclic encoding:
  - sin_day = sin(2π * day / 7)
  - cos_day = cos(2π * day / 7)

- demand_period (custom segmentation):
  - Night, MorningRise, Peak, Decline, LowDemand, Recovery

---

### 4.2 Weather Features
- Temperature imputation using median
- temp_bin (VeryCold → Hot)
- temp_sq (non-linear effect capture)
- weather_temp interaction feature

---

### 4.3 Infrastructure Features
- road_score (Highway > Street > Residential)
- road_capacity = road_score × NumberofLanes
- lane_vehicle_interaction
- Landmarks_bin (Yes/No)
- LargeVehicles_bin (Allowed/Not Allowed)

---

### 4.4 Spatial Features
- geohash (categorical)
- geohash_te (target encoding version used during training)
- weather_mean_demand (training statistic)
- geohash_mean_demand (training statistic)
- geohash_hour_mean (training statistic)
- geohash_weather_mean (training statistic)

---

## 5. Model Details

### Model Used: LightGBM Regressor

### Objective:
- regression

### Metric:
- RMSE (optimized during training)
- Evaluation metric: R²

---

## 6. Hyperparameters

```python
params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 128,
    "max_depth": -1,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbosity": -1
}
```

### Key Design Choices:
- Low learning rate for stability
- Large num_leaves for nonlinear modeling
- Regularization via L2 and bagging
- Feature subsampling for generalization

---

## 7. Training Strategy

- GroupKFold used with geohash as grouping variable
- Prevents spatial leakage across train/test splits
- Early stopping used (200 rounds)
- Model trained per fold and averaged

---

## 8. Artifacts Used

Saved artifacts for inference:

- temp_median → missing value imputation
- geohash_te_map → spatial prior knowledge
- weather_mean_demand → weather baseline behavior
- geohash_hour_mean → temporal-location interaction
- geohash_weather_mean → spatial-weather interaction

---

## 9. Inference Pipeline

### Steps:

1. Clean input data
2. Extract time features
3. Apply cyclic transformations
4. Generate demand_period and temp_bin
5. Encode categorical variables
6. Apply artifact-based mappings
7. Construct feature matrix
8. Predict using trained LightGBM model

---

## 10. How to Reproduce Results

### Step 1: Load artifacts and model

```python
import joblib

model = joblib.load("model.pkl")
artifacts = joblib.load("artifacts.pkl")
```

---

### Step 2: Initialize pipeline

```python
pipeline = DemandFeaturePipeline(artifacts)
model_wrapper = DemandModel(model, pipeline, features)
```

---

### Step 3: Prepare test data

Ensure test data contains:

- geohash
- timestamp
- day
- Weather
- Temperature
- RoadType
- NumberofLanes
- Landmarks
- LargeVehicles

---

### Step 4: Predict

```python
predictions = model_wrapper.predict(test_df)
```

---

## 11. Key Success Factors

- Strong temporal cyclic encoding
- Geohash-based spatial representation
- Interaction features (weather × temperature)
- Infrastructure modeling
- Leakage-safe GroupKFold validation
- Proper artifact-based inference pipeline

---

## 12. Final Outcome

- Model achieves ~97% R² score
- Captures nonlinear spatio-temporal demand patterns
- Robust inference pipeline suitable for deployment

---

## 13. Notes

- Do NOT recompute target-based features during inference
- Ensure categorical consistency between train/test
- Maintain artifact integrity for reproducibility
