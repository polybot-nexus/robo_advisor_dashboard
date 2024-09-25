# Script that evaluates the performance of several ML algorithms on the collected data from
# the high-throughput experiment. After the evaluation return a whisker plot with the comparions
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (  # Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import model_selection
from sklearn.gaussian_process.kernels import WhiteKernel
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

kfold = model_selection.KFold(n_splits=5, random_state=7, shuffle=True)

oect_data = pd.read_csv('datasets/oect_summary_posted_rf__plus_ml_combined.csv')


scaler = MinMaxScaler()
feature_columns = ["coating_on_top.sol_label",	"coating_on_top.substrate_label",	"coating_on_top.vel",	"coating_on_top.T"]
whole_space_scaled=scaler.fit_transform(oect_data[feature_columns].values)
whole_space_scaled = pd.DataFrame(whole_space_scaled, columns = feature_columns)
transcond_scaled=scaler.fit_transform(oect_data['transconductance'].values.reshape(-1, 1))
train_data_scaled = whole_space_scaled.iloc[:21, :]
second_round_data_scaled = whole_space_scaled.iloc[:44, :]

input = train_data_scaled.values #oect_data[oect_data['source'] == 'train'][feature_columns].values
output = transcond_scaled[:21].ravel() #oect_data[oect_data['source'] == 'train'].transconductance.values


def gpregression(Xtrain,Ytrain,Nfeature):
    train_y_avg = np.average(Ytrain)
    noise_estimated = (
              np.std(Ytrain) / 3
          )
    noise_lb = noise_estimated / 4
    noise_ub = noise_estimated * 11
    kernel_krq = (train_y_avg) ** 2 * RBF(length_scale=1)
    kernel_noise = WhiteKernel(
              noise_level=noise_estimated ** 2,
              noise_level_bounds=(noise_lb ** 2, noise_ub ** 2),
          )
    kernel_initial = kernel_krq + kernel_noise
    kernel = kernel_noise + (train_y_avg) ** 2 * RBF(length_scale=1)
    regressor = GaussianProcessRegressor(kernel=kernel, alpha=0, n_restarts_optimizer=5)
    gp = GaussianProcessRegressor(kernel=kernel,  normalize_y=False) #n_restarts_optimizer=40,
    gp.fit(Xtrain, Ytrain)
    return gp

def ML_comparison():
    linear_rmse_list = []
    linear_rmse_list_train=[]

    rmse_list = []
    rmse_list_train = []
    mae_list = []

    gp_rmse_list = []
    gp_rmse_list_train = []

    svr_rmse_list = []
    svr_rmse_list_train = []

    adaboost_rmse_list = []
    adaboost_rmse_list_train = []

    nn_rmse_list = []
    nn_rmse_list_train = []

    for train_index, test_index in kfold.split(input):# #
        print("TRAIN:", train_index, "TEST:", test_index)

        train_shape = input[train_index]
        test_shape = input[test_index]

        print(train_shape.shape, test_shape.shape)
        print(test_index)


        linear_regressor = LinearRegression()
        #XGBRegressor(objective="reg:squarederror", eta=0.1, subsample=0.2, colsample_bytree=0.5, random_state=1, alpha=0.5)
        linear_regressor.fit(input[train_index], output[train_index])
        linear_train_predictions = linear_regressor.predict(input[train_index])
        linear_test_predictions = linear_regressor.predict(input[test_index])
        linear_rmse_list.append(np.sqrt(metrics.mean_squared_error(output[test_index], linear_test_predictions)))
        linear_rmse_list_train.append(np.sqrt(metrics.mean_squared_error(output[train_index], linear_train_predictions)))

        rf_regressor = RandomForestRegressor( n_estimators=2000, random_state=0) #,  max_depth=20
        #rf_regressor = RandomForestRegressor( n_estimators=5000,  max_depth=10, random_state=888) #RandomForestRegressor(n_estimators=200, random_state=0)
        rf_regressor.fit(input[train_index], output[train_index])
        rf_test_predictions = rf_regressor.predict(input[test_index])
        rf_train_predictions = rf_regressor.predict(input[train_index])

        mae_list.append(np.sqrt(metrics.mean_absolute_error(output[test_index], rf_test_predictions)))
        rmse_list.append(np.sqrt(metrics.mean_squared_error(output[test_index], rf_test_predictions)))
        # get train rmse
        rmse_list_train.append(np.sqrt(metrics.mean_squared_error(output[train_index], rf_train_predictions)))

        print('RF Cylce done')

        #GP
        #kernel =  C(0.1, (1e-5, 1e2)) * RBF(100, (1e-3, 1e5))+ RBF(12, (1e-3, 1e5)) +RBF(1, (1e-3, 1e3))

        gp = gpregression(input[train_index], output[train_index],4)
        gp_test_predictions, MSE = gp.predict(input[test_index], return_std=True)
        gp_test_predictions_train, MSE_train = gp.predict(input[train_index], return_std=True)

        gp_rmse_list.append(np.sqrt(metrics.mean_squared_error(output[test_index], gp_test_predictions)))
        gp_rmse_list_train.append(np.sqrt(metrics.mean_squared_error(output[train_index], gp_test_predictions_train)))

        print('GP Cylce done')

        svr = SVR(kernel='rbf')
        svr.fit(input[train_index], output[train_index])

        svr_test_predictions = svr.predict(input[test_index])
        svr_test_predictions_train = svr.predict(input[train_index])

        svr_rmse_list.append(np.sqrt(metrics.mean_squared_error(output[test_index], svr_test_predictions)))
        svr_rmse_list_train.append(np.sqrt(metrics.mean_squared_error(output[train_index], svr_test_predictions_train)))


        print('SVR Cylce done')
        adaboost = AdaBoostRegressor(random_state=0, n_estimators=1000)
        adaboost.fit(input[train_index], output[train_index])

        adaboost_test_predictions = adaboost.predict(input[test_index])
        adaboost_test_predictions_train = adaboost.predict(input[train_index])

        adaboost_rmse_list.append(np.sqrt(metrics.mean_squared_error(output[test_index], adaboost_test_predictions)))
        adaboost_rmse_list_train.append(np.sqrt(metrics.mean_squared_error(output[train_index], adaboost_test_predictions_train)))

        print('adaboost Cylce done')

        # NN
        def build_model():
            tf.random.set_seed(0)
            input_dimension=4
            model = keras.Sequential(name="my_sequential")
            model.add(layers.Dense(4, activation="relu", name="layer1"))
            model.add(layers.Dense(2, activation="relu", name="layer2"))
            model.add(layers.Dense(1, name="layer3"))
            optimizer = tf.keras.optimizers.RMSprop(0.01)
            model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
            return model

        model = build_model()
        EPOCHS = 1000
        history = model.fit(input[train_index], output[train_index],epochs=EPOCHS, validation_split = 0.3, verbose=0)#,callbacks=[tfdocs.modeling.EpochDots()])
        nn_test_predictions = model.predict(input[test_index])#.flatten())
        nn_train_predictions = model.predict(input[train_index])#.flatten())


        nn_rmse_list.append(np.sqrt(metrics.mean_squared_error(output[test_index], nn_test_predictions)))
        nn_rmse_list_train.append(np.sqrt(metrics.mean_squared_error(output[train_index], nn_train_predictions)))
        print('NN Cylce done')

        linear_rmse = np.array(linear_rmse_list)
        linear_rmse_average = np.average(linear_rmse)
        linear_rmse_stdev = np.std(linear_rmse)
        print('The Linear Model average RMSE of group cross validations is:',linear_rmse_average, '+-' ,linear_rmse_stdev)

        rmse = np.array(rmse_list)
        rmse_average = np.average(rmse)
        rf_rmse_stdev = np.std(rmse_list)

        print('The Random Forest average RMSE of the group cross validations is:',rmse_average, '+-' ,rf_rmse_stdev)

        gp_rmse = np.array(gp_rmse_list)
        gp_rmse_average = np.average(gp_rmse)
        gp_rmse_stdev = np.std(gp_rmse)

        print('The Gaussian Process average RMSE of group cross validations is:',gp_rmse_average, '+-' ,gp_rmse_stdev)

        nn_rmse = np.array(nn_rmse_list)
        nn_rmse_average = np.average(nn_rmse)
        nn_rmse_stdev = np.std(nn_rmse)

        print('The Neural Net average RMSE of 10 cross validations is:',nn_rmse_average,'+-',nn_std_rmse)

        svr_rmse = np.array(svr_rmse_list)
        svr_rmse_average = np.average(svr_rmse)
        svr_rmse_stdev = np.std(svr_rmse)

        print('The SVR average RMSE of group cross validations is:',svr_rmse_average, '+-' ,svr_rmse_stdev)

        adaboost_rmse = np.array(adaboost_rmse_list)
        adaboost_rmse_average = np.average(adaboost_rmse)
        adaboost_rmse_stdev = np.std(adaboost_rmse)

        print('The adaboost average RMSE of group cross validations is:', adaboost_rmse_average, '+-' ,adaboost_rmse_stdev)

    return linear_rmse_list_train, linear_rmse, rmse_list_train, rmse, gp_rmse_list_train, gp_rmse, svr_rmse_list_train, svr_rmse, adaboost_rmse_list_train, adaboost_rmse, nn_rmse_list_train, nn_rmse


linear_rmse_list_train, linear_rmse, rmse_list_train, rmse, gp_rmse_list_train, gp_rmse, svr_rmse_list_train, svr_rmse, adaboost_rmse_list_train, adaboost_rmse, nn_rmse_list_train, nn_rmse = ML_comparison()
models_data = {
    "Linear Regression": (linear_rmse_list_train, linear_rmse),
    "Random Forest": (rmse_list_train, rmse),
    "Gaussian Process": (gp_rmse_list_train, gp_rmse),
    "SVR": (svr_rmse_list_train, svr_rmse),
    "AdaBoost": (adaboost_rmse_list_train, adaboost_rmse),
    "Neural Net": (nn_rmse_list_train, nn_rmse)
}


def evaluate_ml_models():
    # Simulate evaluating multiple ML models and returning a comparison plot
    fig, axes = plt.subplots(3, 2, figsize=(12, 16), sharex=True, sharey=True)
    axes = axes.flatten()

    # Bar width
    width = 0.35

    for i, (model_name, (train_data, test_data)) in enumerate(models_data.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        x = np.arange(1, len(train_data) + 1)
        rects1 = ax.bar(x - width/2, train_data, width, label='Train_RMSE', color='orange', edgecolor='black')
        rects2 = ax.bar(x + width/2, test_data, width, label='Test_RMSE', color='dodgerblue', edgecolor='black')
        ax.set_title(f'{model_name} RMSE Across CV Folds', fontsize=14, loc='center', pad=-40)
        ax.set_ylim(0, 0.8)  # set a fixed y-limit if desired
        ax.tick_params(axis='both', labelsize=14)

    # Set a common X and Y label
    fig.supxlabel('CV Fold Number', fontsize=14)
    fig.supylabel('RMSE', fontsize=14)

    # Create a single legend for the whole figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=14, ncol=2)

    # Adjust layout to make room for the common X label, Y label, and legend
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    # Remove any unused axes if number of models is less than subplot slots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    return fig