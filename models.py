import numpy as np
import xgboost as xgb
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.model_selection import cross_val_score, RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_absolute_error


def cv_grid_search(model, X_train, y_train, X_val, y_val, params):
    clf = GridSearchCV(model, params, n_jobs=5, 
                cv=RepeatedKFold(n_splits=5, random_state=1), 
                scoring='neg_mean_absolute_error',
                verbose=0, 
                refit=True)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    return clf.best_params_, clf.best_score_

def kfold_check(model, X, y):
    cv = RepeatedKFold(n_splits=5, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv)
    return scores
    

class XGBoost():
    def __init__(self, kwargs) -> None:
        self.model = xgb.XGBRegressor(kwargs=kwargs, enable_categorical=True)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test(self, X_test, y_test):
        preds = self.model.predict(X_test)
        return mean_absolute_error(y_test, preds)
    
    def plot(self, num_trees):
        fig, ax = plt.subplots(figsize=(20,20))
        xgb.plot_tree(self.model, num_trees=num_trees, ax=ax)
        plt.show()

class DenseNN(nn.Module):
    def __init__(self, input_channels, learning_rate):
        super().__init__()
        self.hidden_size = input_channels*2
        self.model = nn.Sequential(
            nn.Linear(input_channels, input_channels),
            nn.ReLU(),
            nn.Linear(input_channels, 4*self.hidden_size),
            nn.ReLU(),
            nn.Linear(4*self.hidden_size, 8*self.hidden_size),
            nn.ReLU(),
            nn.Linear(8*self.hidden_size, 4*self.hidden_size),
            nn.ReLU(),
            nn.Linear(4*self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_func = nn.L1Loss()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)

def nn_train(
    model: nn.Module,
    dataloader: DataLoader,
    num_epoch: int,
) -> list[float]:
    
    epoch_avg_losses = []
    model.train()
    
    for epoch in tqdm(range(num_epoch)):
        epoch_loss_sum = 0

        for X, Y in dataloader:
            output = model(X)
            model.optimizer.zero_grad()
            loss = model.loss_func(output, Y)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            model.optimizer.step()
            epoch_loss_sum += loss.item() * X.shape[0]

        avg_epoch_loss = epoch_loss_sum / len(dataloader.dataset)
        epoch_avg_losses.append(avg_epoch_loss)

    return epoch_avg_losses
        

def nn_test(model: nn.Module, dataloader: DataLoader) -> float:
    loss_sum = 0
    model.eval()
    with torch.no_grad():
        for X, Y in dataloader:
            output = model(X)
            loss = model.loss_func(output, Y)
            loss_sum += loss.item() * X.shape[0]
            
    avg_loss = loss_sum / len(dataloader.dataset)

    return avg_loss
