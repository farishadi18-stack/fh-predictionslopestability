import streamlit as st
import numpy as np
import pandas as pd

st.title(' Prediction Factor of Safety')

st.info('This is app create prediction using soil nailing !!')

# Step 1: Load Data
with st.expander('Data'):
  st.write('Raw Data')
  df = pd.read_csv('https://raw.githubusercontent.com/farishadi18-stack/fh-predictionslopestability/refs/heads/master/new%20treated%20slope.csv')
  df

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import math

# --- Sidebar ---
test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
random_state = st.sidebar.number_input("Random Seed", 0, 1000, 42)

# --- Main ---

    X = df.drop(columns=['Factor_of_Safety'])
    y = df['Factor_of_Safety']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Define models
    models = {
        "Linear Regression": LinearRegression(),
        "Polynomial Regression": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        "SVR": SVR(kernel='rbf'),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Decision Tree": DecisionTreeRegressor(random_state=random_state),
        "ANN": MLPRegressor(hidden_layer_sizes=(64,32), activation='relu', max_iter=1000, random_state=random_state)
    }

    # Cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=random_state)
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
        cv_results[name] = np.mean(scores)
    cv_df = pd.DataFrame(cv_results, index=["R2 Score (CV)"]).T

    # Final test metrics
    final_results = {}
    n, p = X_test.shape[0], X_test.shape[1]
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        final_results[name] = {
            "R2": r2,
            "Adjusted R2": adj_r2,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE (%)": mape
        }

    final_df = pd.DataFrame(final_results).T
    summary_df = cv_df.join(final_df, how="outer")

    st.write("### Model Performance Summary", summary_df)

    # --- Plot Predictions ---
    st.write("### Predictions vs Actuals")
    n_models = len(models)
    cols = 3
    rows = math.ceil(n_models / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = axes.flatten()

    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        axes[i].scatter(y_test, preds, alpha=0.7, label="Predicted")
        axes[i].plot([y_test.min(), y_test.max()],
                     [y_test.min(), y_test.max()],
                     'r--', label="Perfect Fit")
        axes[i].set_title(name)
        axes[i].set_xlabel("Actual FoS")
        axes[i].set_ylabel("Predicted FoS")
        axes[i].legend()
    plt.tight_layout()
    st.pyplot(fig)

    # --- Metric comparison bar chart ---
    st.write("### Comparison of Metrics")
    fig2, axes2 = plt.subplots(2, 3, figsize=(18,10))
    axes2 = axes2.flatten()
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948"]

    for i, metric in enumerate(final_df.columns):
        axes2[i].bar(final_df.index, final_df[metric], color=colors[i], edgecolor="black")
        axes2[i].set_title(metric)
        axes2[i].set_ylabel("Value")
        axes2[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig2)
