import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import merge_data, get_data
from models import cv_grid_search, kfold_check, nn_train, nn_test
from models import XGBoost, DenseNN

import os
os.environ["PATH"] += os.pathsep + "C:/Users/erict/Downloads/windows_10_cmake_Release_Graphviz-11.0.0-win64/Graphviz-11.0.0-win64/bin"

def lasso_test(X, y):
    print("Lasso Test Results:")
    model = Lasso(alpha=0.5, max_iter=10000)
    scores = kfold_check(model, X, y)
    print(f"K-Fold CV Mean Absolute Error: {np.mean(scores)}")

def xgb_test(params, X, y, X_train, y_train, X_test, y_test):
    print("XGBoost Test Results:")
    model = XGBoost(params)
    scores = kfold_check(model.model, X, y)
    print(f"K-Fold CV Mean Absolute Error: {np.mean(scores)}")

    model.train(X_train, y_train)
    print(f"Model test results: {model.test(X_test, y_test)}")

    model.plot(num_trees=1)

def rf_test(model, X, y, X_train, y_train, X_test, y_test):
    print("RF Test Results:")
    scores = kfold_check(model, X, y)
    print(f"K-Fold CV Mean Absolute Error: {np.mean(scores)}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Model test results: {mean_absolute_error(y_test, preds)}")


def main(model_choice, grid=False):
    # search for best hyperparameters for our XGBoost model
    if grid:
        xgb_grid_search_params = { 
                'max_depth': [3, 5, 8],
                'learning_rate': [0.001, 0.01, 0.1],
                'gamma':[0, 0.25, 0.5],
                'n_estimators': [500, 1000],
                'subsample':[0.8, 0.9],
                'colsample_bytree':[0.5],
                'early_stopping_rounds': [10],
        }
        best_params, best_score = cv_grid_search(xgb.XGBRegressor(enable_categorical=True), 
                                                 X_train, y_train, X_test, y_test, xgb_grid_search_params)
        print(best_params, best_score)

    if model_choice == "lasso":
        X, y = get_data("LEGO_Data/merged.csv", (1975, 2023), xgb_=False)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        lasso_test(X, y)

    elif model_choice == "xgb":
        X, y = get_data("LEGO_Data/merged.csv", (2016, 2021))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        params = {
            'colsample_bytree': 0.5, 
            'early_stopping_rounds': 10, 
            'gamma': 0.25, 
            'learning_rate': 0.01, 
            'max_depth': 5, 
            'n_estimators': 500, 
            'subsample': 0.8
        }
        xgb_test(params, X, y, X_train, y_train, X_test, y_test)

    elif model_choice == "rf":
        X, y = get_data("LEGO_Data/merged.csv", (1975, 2020), xgb_=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor(max_depth=7, n_estimators=1000, min_samples_split=10)
        rf_test(model, X, y, X_train, y_train, X_test, y_test)

    elif model_choice == "dense":
        X, y = get_data("LEGO_Data/merged.csv", (1975, 2020), xgb_=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        dense_model = DenseNN(input_channels=X.shape[1], learning_rate=0.001)

        train = TensorDataset(torch.Tensor(X_train.values), torch.Tensor(y_train.values))
        test = TensorDataset(torch.Tensor(X_test.values), torch.Tensor(y_test.values))
        dataloader_train = DataLoader(train, batch_size=64, shuffle=True)
        dataloader_test = DataLoader(test, batch_size=32, shuffle=True)

        epoch_loss = nn_train(dense_model, dataloader_train, num_epoch=25)
        plt.plot(epoch_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        print(nn_test(dense_model, dataloader_test))

if __name__ == '__main__':
    #merge_data("LEGO_Data/sets.csv", "LEGO_Data/lego_final_data.xlsx")
    main("xgb", grid=False)