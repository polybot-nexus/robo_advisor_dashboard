import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import plotly.graph_objs as go
from joblib import dump, load
import os
import shap
import plotly.express as px


# Gaussian Process Regressor
def gp_regression(Xtrain, Ytrain):
    train_y_avg = np.average(Ytrain)
    noise_estimated = np.std(Ytrain) / 3
    kernel = train_y_avg**2 * RBF(length_scale=1) + WhiteKernel(noise_level=noise_estimated**2)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=False)
    gp.fit(Xtrain, Ytrain)
    return gp

# Neural Network Model
def build_nn_model():
    model = keras.Sequential([
        layers.Dense(4, activation="relu", input_shape=(4,)),
        layers.Dense(2, activation="relu"),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model

# Perform evaluation using different models
def evaluate_ml_models(oect_data):
    scaler = MinMaxScaler()
    feature_columns = ["coating_on_top.sol_label", "coating_on_top.substrate_label", "coating_on_top.vel", "coating_on_top.T"]
    whole_space_scaled = scaler.fit_transform(oect_data[feature_columns].values)
    whole_space_scaled = pd.DataFrame(whole_space_scaled, columns=feature_columns)
    transcond_scaled = scaler.fit_transform(oect_data['transconductance'].values.reshape(-1, 1))

    train_data_scaled = whole_space_scaled.iloc[:, :]
    input = train_data_scaled.values
    output = transcond_scaled[:].ravel()
    results = {
        "Gaussian Process": {"train_rmse": [], "test_rmse": [], "test_rmse_average": [], "test_rmse_std": []},
        "Linear Regression": {"train_rmse": [], "test_rmse": [], "test_rmse_average": [], "test_rmse_std": []},
        "SVR": {"train_rmse": [], "test_rmse": [], "test_rmse_average": [], "test_rmse_std": []},
        "Neural Net": {"train_rmse": [], "test_rmse": [], "test_rmse_average": [], "test_rmse_std": []},
        "Random Forest": {"train_rmse": [], "test_rmse": [], "test_rmse_average": [], "test_rmse_std": []},         
        "AdaBoost": {"train_rmse": [], "test_rmse": [], "test_rmse_average": [], "test_rmse_std": []}        
    }
    kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)
    for train_index, test_index in kfold.split(input):
        X_train, X_test = input[train_index], input[test_index]
        y_train, y_test = output[train_index], output[test_index]

        # Gaussian Process
        gp = gp_regression(X_train, y_train)
        results["Gaussian Process"]["train_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_train, gp.predict(X_train))))
        results["Gaussian Process"]["test_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_test, gp.predict(X_test))))
        
        # Linear Regression
        linear_reg = LinearRegression().fit(X_train, y_train)
        results["Linear Regression"]["train_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_train, linear_reg.predict(X_train))))
        results["Linear Regression"]["test_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_test, linear_reg.predict(X_test))))
        
        # Support Vector Regressor (SVR)
        svr = SVR(kernel='rbf').fit(X_train, y_train)
        results["SVR"]["train_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_train, svr.predict(X_train))))
        results["SVR"]["test_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_test, svr.predict(X_test))))
        
        # Neural Network
        if os.path.exists('ml_model_weights/nn_model_weights.h5'):
            print("Loading saved weights...")
            nn_model = build_nn_model()  # Ensure the model architecture is the same
            nn_model.load_weights('ml_model_weights/nn_model_weights.h5')
            print("Weights loaded successfully. Skipping training.")
        else:
            print("Weights not found. Training the model...")
            nn_model = build_nn_model()
            nn_model.fit(X_train, y_train, epochs=1000, verbose=0)
            nn_model.save_weights('ml_model_weights/nn_model_weights.h5')
            print("Training complete and weights saved.")

        results["Neural Net"]["train_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_train, nn_model.predict(X_train))))
        results["Neural Net"]["test_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_test, nn_model.predict(X_test))))
        
        # Random Forest
        rf_reg = RandomForestRegressor(n_estimators=2000, random_state=0).fit(X_train, y_train)
        results["Random Forest"]["train_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_train, rf_reg.predict(X_train))))
        results["Random Forest"]["test_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_test, rf_reg.predict(X_test))))

        # AdaBoost Regressor
        adaboost = AdaBoostRegressor(random_state=0, n_estimators=1000).fit(X_train, y_train)
        results["AdaBoost"]["train_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_train, adaboost.predict(X_train))))
        results["AdaBoost"]["test_rmse"].append(
            np.sqrt(metrics.mean_squared_error(y_test, adaboost.predict(X_test))))


    models = list(results.keys())
    for model in models:
        results[model]["train_rmse_average"] =  np.average(results[model]["train_rmse"])
        results[model]["test_rmse_average"] =  np.average(results[model]["test_rmse"])
        results[model]["test_rmse_std"] =  np.std(results[model]["test_rmse"])
    return results


def plot_ml_comparison(results):
    """ Function to plot model comparison"""
    models = list(results.keys())
    fig, axes = plt.subplots(3, 2, figsize=(12, 16), sharex=True, sharey=True)
    axes = axes.flatten()
    width = 0.35

    for i, model in enumerate(models):
        ax = axes[i]
        train_rmse = results[model]["train_rmse"]
        test_rmse = results[model]["test_rmse"]
        x = np.arange(1, len(train_rmse) + 1)
        ax.bar(x - width/2, train_rmse, width, label='Train RMSE', color='orange', edgecolor='black')
        ax.bar(x + width/2, test_rmse, width, label='Test RMSE', color='dodgerblue', edgecolor='black')
        ax.set_title(f'{model} RMSE', fontsize=14)
        ax.set_ylim(0, 0.8)
        ax.tick_params(axis='both', labelsize=14)

    fig.supxlabel('CV Fold Number', fontsize=14)
    fig.supylabel('RMSE', fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # plt.show()
    return fig


def plot_whiskerplots(results):
    """
    Plots RMSE values with uncertainties for different models using Plotly.

    Parameters:
    - results (dict): A dictionary with model names as keys and tuples of (rmse_average, rmse_stdev).

    Returns:
    - fig: Plotly figure object with the plot.
    """
    models = list(results.keys())
    values = [results[model]["test_rmse_average"] for model in models]
    uncertainties = [results[model]["test_rmse_std"] for model in models]

    lower_bounds = [v - u for v, u in zip(values, uncertainties)]
    upper_bounds = [v + u for v, u in zip(values, uncertainties)]

    # Create figure
    fig = go.Figure()

    # Add whisker plots for each model
    for i, model in enumerate(models):
        fig.add_trace(go.Box(
            y=[lower_bounds[i], values[i], upper_bounds[i]],  # RMSE with uncertainty
            name=model,
            boxmean='sd',  # Display standard deviation
            showlegend=False
        ))

    # Set the layout for the figure
    fig.update_layout(
        title='Whisker plot of the RMSE values',
        xaxis_title='Models',
        yaxis_title='RMSE',
        xaxis_tickangle=-45,
        height=600,
        width=1000
    )

    return fig


def shapley_analysis_plotly(oect_data):
    scaler = MinMaxScaler()
    feature_columns = ["coating_on_top.sol_label", "coating_on_top.substrate_label", "coating_on_top.vel", "coating_on_top.T"]
    whole_space_scaled = scaler.fit_transform(oect_data[feature_columns].values)
    whole_space_scaled = pd.DataFrame(whole_space_scaled, columns=feature_columns)
    transcond_scaled = scaler.fit_transform(oect_data['transconductance'].values.reshape(-1, 1))

    mdl = RandomForestRegressor(n_estimators=1000,  max_depth=20, random_state=888 ) #max_depth=12, random_state=888)
    X, y = whole_space_scaled, transcond_scaled[:].ravel()
    # print('y', len(y))
    mdl.fit(X, y)
    explainer = shap.TreeExplainer(mdl, feature_perturbation='tree_path_dependent')
    shap_values = explainer.shap_values(X)
    shap_df = pd.DataFrame(shap_values, columns=['OMIEC concentration', 'Substrate features', 'Coating speed', 'Coating temperature'])    
    shap_range = (shap_df.max()).sort_values(ascending=False)    
    feature_df = pd.DataFrame(X.values, columns=['OMIEC concentration', 'Substrate features', 'Coating speed', 'Coating temperature'])
    shap_long = shap_df.melt(var_name='Feature', value_name='SHAP Value')
    feature_long = feature_df.melt(var_name='Feature', value_name='Feature Value')
    combined_df = pd.concat([shap_long, feature_long['Feature Value']], axis=1)
    combined_df['Feature'] = pd.Categorical(combined_df['Feature'], categories=shap_range.index, ordered=True)
    combined_df = combined_df.sort_values(by='Feature', ascending=False)

    fig = px.scatter(
        combined_df,         
        x='SHAP Value', 
        y='Feature',         
        color='Feature Value',
        labels={'SHAP Value': 'SHAP Value (Impact on Model Output)', 'Feature': 'Feature (Ranked by Extremes)'},
        color_continuous_scale='Viridis',
        hover_data=['SHAP Value', 'Feature Value']
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(245, 245, 245, 0.8)',  # Light gray background
        paper_bgcolor='rgba(255, 255, 255, 0.8)',  # Almost white paper background
        xaxis=dict(
            title="SHAP Value",
            title_font=dict(size=20),  # Increased title font size
            tickfont=dict(size=30),    # Increased tick label font size
            gridcolor='white'          # White grid lines
        ),
        yaxis=dict(
            title="Feature",
            title_font=dict(size=20),  # Increased title font size
            tickfont=dict(size=30),    # Increased tick label font size
            gridcolor='white'          # White grid lines
        ),
        coloraxis_colorbar=dict(
            title="Feature Value",
            title_font=dict(size=18),  # Increased colorbar title font size
            tickfont=dict(size=16)     # Increased colorbar tick font size
        ),
        margin=dict(l=50, r=50, t=20, b=40),
        height=500,
        width=800
    )
    
    fig.update_traces(
        marker=dict(
            size=20,
            opacity=0.8
        )
    )
    
    fig.add_vline(
        x=0,
        line=dict(color="black", width=2, dash="dash"),
    )
    
    return fig

# results = evaluate_ml_models()
# print(results)
# plot_ml_comparison(results)
# plot_whiskerplots(results)
# shapley_analysis_plotly()

import json
# Load and scale the data
oect_data = pd.read_csv('datasets/oect_summary_posted_rf__plus_ml_combined.csv')
oect_data['coating_on_top.sol_label'] = oect_data['coating_on_top.sol_label'].map(lambda x: x.lstrip('mg/ml').rstrip('mg/ml'))
oect_data['coating_on_top.substrate_label'] = oect_data['coating_on_top.substrate_label'].map(lambda x: x.lstrip('nm').rstrip('nm'))
oect_data['coating_on_top.sol_label'] = pd.to_numeric(oect_data['coating_on_top.sol_label'])
oect_data['coating_on_top.substrate_label'] = pd.to_numeric(oect_data['coating_on_top.substrate_label'])
oect_data['coating_on_top.vel'] = pd.to_numeric(oect_data['coating_on_top.vel'])
oect_data['coating_on_top.T'] = pd.to_numeric(oect_data['coating_on_top.T'])

def precompute_results(start_idx,  end_idx):
    
    results = evaluate_ml_models(oect_data.iloc[start_idx:end_idx])
    with open(f"ml_model_weights/results_{start_idx}_{end_idx}.json", 'w') as f:
        json.dump(results, f)


# precompute_results(0, oect_data.shape[0])

# for i in range(20, oect_data.shape[0]):
#    precompute_results(0, i)



# import plotly.io as pio

# Precompute and save the plot
# fig = shapley_analysis_plotly()
# pio.write_json(fig, "shap_plot.json")