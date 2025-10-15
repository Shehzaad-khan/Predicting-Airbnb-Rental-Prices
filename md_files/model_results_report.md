# 🏠 Airbnb Price Prediction — Model Results

This document summarizes the performance of all machine learning models applied to the **Airbnb Price Prediction** project.  
Each model was trained and evaluated on the **processed NYC Airbnb dataset**.

---

## 📊 Model Evaluation Results

### **Linear Regression**
- **R²:** 0.299  
- **RMSE:** $250.09  

### **Random Forest**
- **R²:** 0.476  
- **RMSE:** $216.38  

### **Support Vector Regressor (SVR)**
- **R²:** 0.216  
- **RMSE:** $264.58  

### **K-Means Clustering (Price Segmentation)**
| Cluster | Mean Price ($) |
|----------|----------------|
| 0 | 458.59 |
| 1 | 190.50 |
| 2 | 305.63 |
| 3 | 161.91 |
| 4 | 180.20 |
- **R²:** 0.3427  
- **RMSE:** $285.44 

*(Used for segmentation, not prediction)*

### **Neural Network**
- **R²:** 0.3427  
- **RMSE:** $242.23  

---

## 🏆 Summary of Findings

- **Best Performing Model:** 🎯 **Random Forest**
  - Highest **R² (0.476)** and lowest **RMSE ($216.38)**  
  - Demonstrates strong predictive accuracy for Airbnb price prediction.

- **Neural Network:**  
  - Achieved moderate performance (**R² = 0.34**),  
  - Can be further improved through tuning (e.g., more epochs, different learning rate, or deeper architecture).

- **Linear Regression & SVR:**  
  - Performed decently but limited in handling complex, non-linear relationships.

- **K-Means Clustering:**  
  - Not a predictive model, but useful for identifying **price-based customer segments** and **market grouping patterns**.

---

## 🧠 Key Takeaways

- **Random Forest** is the most reliable model for predicting Airbnb listing prices.  
- **K-Means** helps understand **market segmentation** and **pricing tiers**.  
- **Neural Networks** show potential for improvement with hyperparameter optimization.  
- Overall, the project demonstrates how different machine learning models perform on structured real-world pricing data.

---

✅ *All results are based on the final processed dataset used in the project notebooks.*

