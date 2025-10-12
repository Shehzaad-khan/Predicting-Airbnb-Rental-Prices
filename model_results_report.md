# Airbnb Price Prediction Model Results

This file records the performance of all models run in the notebook:

## Model Outputs

**Linear Regression**
- R²: 0.299
- RMSE: $250.09

**Random Forest**
- R²: 0.476
- RMSE: $216.38

**SVR**
- R²: 0.216
- RMSE: $264.58

**KMeans Clustering (mean price per segment)**
- Cluster 0: $458.59
- Cluster 1: $190.50
- Cluster 2: $305.63
- Cluster 3: $161.91
- Cluster 4: $180.20

**Neural Network**
- R²: -0.538
- RMSE: $370.59

## Summary
- **Best model:** Random Forest (highest R², lowest RMSE)
- **KMeans:** Useful for price segmentation, not prediction
- **Neural Network:** May require further tuning for better results

All results are based on the processed NYC Airbnb dataset and the notebook's final cell outputs.