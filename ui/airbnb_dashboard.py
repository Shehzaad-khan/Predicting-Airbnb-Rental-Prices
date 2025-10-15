import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

st.set_page_config(page_title="Airbnb Price Prediction Dashboard", layout="wide")
st.title("🏠 Airbnb Price Prediction Dashboard")
st.markdown("""
This interactive dashboard lets you explore and compare different machine learning models for predicting Airbnb rental prices. Upload your processed data, select a model, and view results instantly — including predictions for custom user inputs!
""")

uploaded_train = st.file_uploader("Upload Training Data (airbnb_train_processed.csv)", type=["csv"])
uploaded_test = st.file_uploader("Upload Test Data (airbnb_test_processed.csv)", type=["csv"])

if uploaded_train and uploaded_test:
    train = pd.read_csv(uploaded_train)
    test = pd.read_csv(uploaded_test)
    X_train = train.drop('price', axis=1)
    y_train = train['price']
    X_test = test.drop('price', axis=1)
    y_test = test['price']

    st.header("🔢 Enter Listing Features For Custom Prediction")
    feature_inputs = {}
    for col in X_train.columns:
        if train[col].dtype in ['float64', 'int64']:
            # No min_value; allows negative input
            feature_inputs[col] = st.number_input(
                f"{col}",
                value=float(train[col].mean())
            )
        else:
            feature_inputs[col] = st.text_input(f"{col}", value=str(train[col].mode()[0]))
    user_features = pd.DataFrame([feature_inputs])
    user_features = user_features[X_train.columns]  # Align column order for all models

    st.sidebar.header("Select Model")
    model_choice = st.sidebar.selectbox(
        "Model",
        ["Linear Regression", "Random Forest", "SVR", "KMeans Clustering", "Neural Network"]
    )

    if model_choice == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        st.subheader("Linear Regression Results")
        st.write(f"R²: {r2:.3f}")
        st.write(f"RMSE: ${rmse:.2f}")
        st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": pred}))
        pred_single = model.predict(user_features)[0]
        st.write(f"🔮 Predicted Price (Input): ${pred_single:.2f}")

    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        st.subheader("Random Forest Results")
        st.write(f"R²: {r2:.3f}")
        st.write(f"RMSE: ${rmse:.2f}")
        st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": pred}))
        user_features = user_features[X_train.columns]
        pred_single = model.predict(user_features)[0]
        st.write(f"🔮 Predicted Price (Input): ${pred_single:.2f}")

    elif model_choice == "SVR":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = SVR(kernel='rbf', C=100, gamma=0.1)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** 0.5
        st.subheader("SVR Results")
        st.write(f"R²: {r2:.3f}")
        st.write(f"RMSE: ${rmse:.2f}")
        st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": pred}))
        user_features = user_features[X_train.columns]
        user_features_scaled = scaler.transform(user_features)
        pred_single = model.predict(user_features_scaled)[0]
        st.write(f"🔮 Predicted Price (Input): ${pred_single:.2f}")

    elif model_choice == "KMeans Clustering":
        kmeans = KMeans(n_clusters=5, random_state=42)
        clusters = kmeans.fit_predict(X_train)
        train['price_cluster'] = clusters
        cluster_means = train.groupby('price_cluster')['price'].mean()
        st.subheader("KMeans Price Segments")
        st.write(cluster_means)
        st.bar_chart(cluster_means)
        user_features = user_features[X_train.columns]
        cluster_pred = kmeans.predict(user_features)[0]
        st.write(f"🔮 Input Listing falls in Price Cluster: {cluster_pred} (Avg. Price: ${cluster_means[cluster_pred]:.2f})")

    elif model_choice == "Neural Network":
        class AirbnbNN(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1)
                )
            def forward(self, x):
                return self.net(x).squeeze()
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
        X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_t = torch.tensor(y_test.values, dtype=torch.float32)
        model = AirbnbNN(X_train.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            output = model(X_train_t)
            loss = loss_fn(output, y_train_t)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = model(X_test_t).numpy()
            r2 = r2_score(y_test_t.numpy(), pred)
            rmse = mean_squared_error(y_test_t.numpy(), pred) ** 0.5
        st.subheader("Neural Network Results")
        st.write(f"R²: {r2:.3f}")
        st.write(f"RMSE: ${rmse:.2f}")
        st.line_chart(pd.DataFrame({"Actual": y_test, "Predicted": pred}))
        user_features = user_features[X_train.columns]
        user_features_t = torch.tensor(user_features.values, dtype=torch.float32)
        with torch.no_grad():
            pred_single = model(user_features_t).item()
        st.write(f"🔮 Predicted Price (Input): ${pred_single:.2f}")

else:
    st.info("Please upload both training and test processed CSV files to begin.")
