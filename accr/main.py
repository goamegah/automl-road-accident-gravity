import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler, StandardScaler  # Autorisé comme preprocessing si besoin
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from itertools import product

# 1. Stratified train-test split (custom)
def stratified_split(X, y, test_size=0.2):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    n_test_samples = (class_counts * test_size).astype(int)

    test_indices, train_indices = [], []
    for cls, n_samples in zip(unique_classes, n_test_samples):
        indices = np.where(y == cls)[0]
        np.random.shuffle(indices)
        test_indices.extend(indices[:n_samples])
        train_indices.extend(indices[n_samples:])

    return X.iloc[train_indices], X.iloc[test_indices], y[train_indices], y[test_indices]

# 2. Custom metrics
def f1_score_custom(y_true, y_pred):
    unique_classes = np.unique(y_true)
    f1_scores = []
    for cls in unique_classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    return np.mean(f1_scores)

def confusion_matrix_custom(y_true, y_pred):
    unique_classes = np.unique(y_true)
    n_classes = len(unique_classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for i, cls_true in enumerate(unique_classes):
        for j, cls_pred in enumerate(unique_classes):
            matrix[i, j] = np.sum((y_true == cls_true) & (y_pred == cls_pred))
    return matrix

# 3. Preprocessing
train_ds = pd.read_csv('./dataset/preprocessed_data_v4.csv')
X, y = train_ds.drop(columns=['grav']), train_ds['grav'] - 1
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test = stratified_split(X_scaled, y)

# 4. Model Training and Hyperparameter Tuning
def train_xgb(X_train, y_train, X_val, y_val, params):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, evals=[(dval, 'validation')], verbose_eval=0)
    return model

def train_lightgbm(X_train, y_train, X_val, y_val, params):
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(params, train_data, valid_sets=[val_data]) 
    return model

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.values
        self.y = y.values

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def train_mlp(X_train, y_train, X_val, y_val, params):
    train_data = TabularDataset(X_train, y_train)
    val_data = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_data, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=params['batch_size'], shuffle=False)

    model = MLPModel(X_train.shape[1], params['hidden_dim'], len(np.unique(y_train)))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    for epoch in range(params['epochs']):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    return model

# Évaluer les modèles
def evaluate_model(model, X, y, model_type):
    if model_type == 'xgb':
        dmatrix = xgb.DMatrix(X)
        y_pred = model.predict(dmatrix)
    elif model_type == 'lgb':
        y_pred = model.predict(X).argmax(axis=1)
    elif model_type == 'mlp':
        with torch.no_grad():
            y_pred = torch.argmax(model(torch.tensor(X.values, dtype=torch.float32)), axis=1).numpy()
    f1 = f1_score_custom(y, y_pred)
    matrix = confusion_matrix_custom(y, y_pred)
    return f1, matrix

# Grid search personnalisé
def custom_grid_search(train_fn, X_train, y_train, X_val, y_val, param_grid, model_type):
    """
    Paramètres :
        - train_fn : fonction d'entraînement du modèle
        - X_train, y_train : données d'entraînement
        - X_val, y_val : données de validation
        - param_grid : dictionnaire d'hyperparamètres
        - model_type : type du modèle ('xgb', 'lgb', 'mlp')
    Retourne :
        - Le meilleur modèle et ses hyperparamètres
    """
    best_model = None
    best_score = -np.inf
    best_params = None

    # Générer toutes les combinaisons d'hyperparamètres
    keys, values = zip(*param_grid.items())
    for combination in product(*values):
        params = dict(zip(keys, combination))
        print(f"Testing {model_type} with params: {params}")

        # Entraîner le modèle avec les hyperparamètres courants
        model = train_fn(X_train, y_train, X_val, y_val, params)

        # Évaluer les performances
        f1, _ = evaluate_model(model, X_val, y_val, model_type)

        # Si le modèle est meilleur, le sauvegarder
        if f1 > best_score:
            best_model = model
            best_score = f1
            best_params = params

    print(f"Best {model_type} model: F1 score = {best_score}, params = {best_params}")
    return best_model, best_params

# Hyperparameter grids
xgb_param_grid = {
    'max_depth': [35, 40, 45, 50, 55, 60],
    'eta': [0.1, 0.3, 0.5, 0.7, 0.9],
    'objective': ['multi:softmax'],
    'num_class': [len(np.unique(y))]
}

lgb_param_grid = {
    'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.5],
    'num_leaves': [128, 256, 512, 1024],
    'objective': ['multiclass'],
    'num_class': [len(np.unique(y))]
}

mlp_param_grid = {
    'hidden_dim': [16],
    'lr': [0.001],
    'batch_size': [32],
    'epochs': [10, 50]
}

# Appliquer le grid search à chaque modèle
xgb_best_model, xgb_best_params = custom_grid_search(train_xgb, X_train, y_train, X_test, y_test, xgb_param_grid, 'xgb')
lgb_best_model, lgb_best_params = custom_grid_search(train_lightgbm, X_train, y_train, X_test, y_test, lgb_param_grid, 'lgb')
mlp_best_model, mlp_best_params = custom_grid_search(train_mlp, X_train, y_train, X_test, y_test, mlp_param_grid, 'mlp')

f1_xgb, matrix_xgb = evaluate_model(xgb_best_model, X_test, y_test, 'xgb')
f1_lgb, matrix_lgb = evaluate_model(lgb_best_model, X_test, y_test, 'lgb')
f1_mlp, matrix_mlp = evaluate_model(mlp_best_model, X_test, y_test, 'mlp')

# Sélectionner le meilleur modèle
best_model_type, best_model, best_params, best_f1 = max(
    [('xgb', xgb_best_model, xgb_best_params, f1_xgb),
     ('lgb', lgb_best_model, lgb_best_params, f1_lgb),
     ('mlp', mlp_best_model, mlp_best_params, f1_mlp)],
    key=lambda x: x[3]
)

print(f"Final best model: {best_model_type} with F1 score = {best_f1} and params = {best_params}")
