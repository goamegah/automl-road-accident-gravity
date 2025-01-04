import numpy as np
import pandas as pd

# Fonction de split stratifié
# Remplacement de scikit-learn avec une implémentation maison
def stratified_train_test_split(X, y, test_size, random_state):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    proportions = class_counts / class_counts.sum()
    indices = np.arange(len(y))
    np.random.seed(random_state)

    train_indices = []
    test_indices = []
    for class_value, proportion in zip(unique_classes, proportions):
        class_indices = indices[y == class_value]
        np.random.shuffle(class_indices)
        split_idx = int(len(class_indices) * (1 - test_size))
        train_indices.extend(class_indices[:split_idx])
        test_indices.extend(class_indices[split_idx:])

    return (
        X.iloc[train_indices], X.iloc[test_indices],
        y.iloc[train_indices], y.iloc[test_indices]
    )

# Fonction de normalisation min-max manuelle
def min_max_normalize(dataframe, columns):
    for col in columns:
        min_val = dataframe[col].min()
        max_val = dataframe[col].max()
        dataframe[col] = (dataframe[col] - min_val) / (max_val - min_val)
    return dataframe

# Chargement des données
file_path = './dataset/cleaned_data.csv'
dataset = pd.read_csv(file_path)

# Traitement des valeurs manquantes
dataset = dataset.dropna(subset=['grav'])
missing_threshold = 0.7 * len(dataset)
columns_to_drop = [col for col in dataset.columns if dataset[col].isnull().sum() > missing_threshold]
dataset = dataset.drop(columns=columns_to_drop)

# Encodage des colonnes catégoriques
categorical_columns = dataset.select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_columns:
    dataset[col] = dataset[col].astype('category').cat.codes

# Normalisation des colonnes numériques
numerical_columns = dataset.select_dtypes(include=['float64', 'int64']).columns.tolist()
numerical_columns.remove('grav')  # Exclure la cible
dataset = min_max_normalize(dataset, numerical_columns)

# Séparation des données en X et y
X = dataset.drop(columns=['grav'])
y = dataset['grav'].astype(int)
y = y.replace({1: 0, 2: 1, 3: 2, 4: 3})

# Split stratifié en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, random_state=42)

# Information sur les données préparées
print("Forme des données d'entraînement:", X_train.shape)
print("Forme des données de test:", X_test.shape)
print("Distribution des classes dans l'ensemble d'entraînement:", y_train.value_counts(normalize=True))
print("Distribution des classes dans l'ensemble de test:", y_test.value_counts(normalize=True))


print(np.unique(y))
