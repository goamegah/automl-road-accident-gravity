import pandas as pd
import numpy as np
import functools as ft

class PreProcessing:
    def ordinal_encode(self, df, l_order):
        """
        Custom implementation of ordinal encoding.
        """
        mapping = {col: {val: idx for idx, val in enumerate(l_order)} for col in df.columns}
        encoded_df = df.apply(lambda col: col.map(mapping[col.name]) if col.name in mapping else col)
        return encoded_df, mapping

    def one_hot_encode(self, df, drop="first"):
        """
        Custom one-hot encoding.
        """
        encoded_df = pd.get_dummies(df, drop_first=(drop == "first"))
        return encoded_df, None  # Return None for mapping, as it's not required for one-hot encoding

    def dropna(self, df, subset=None):
        """
        Custom dropna method.
        """
        return df.dropna(subset=subset)

    def imputation(self, df, strategy="mean", fill_value=None):
        """
        Impute missing values based on the strategy.
        """
        if strategy == "mean":
            return df.fillna(df.mean())
        elif strategy == "median":
            return df.fillna(df.median())
        elif strategy == "mode":
            return df.fillna(df.mode().iloc[0])
        elif fill_value is not None:
            return df.fillna(fill_value)
        else:
            raise ValueError("Invalid strategy or fill_value.")

    def knn_imputation(self, df, k=5):
        """
        Custom KNN imputation (simple placeholder for k-nearest neighbors).
        """
        from scipy.spatial.distance import cdist
        df_imputed = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_cols]

        # Placeholder: Impute using simple mean of the nearest neighbors.
        for col in df_numeric.columns:
            missing = df_numeric[col].isna()
            if missing.any():
                distances = cdist(df_numeric.fillna(df_numeric.mean()), df_numeric[~missing].T, 'euclidean')
                nearest_neighbors = np.argsort(distances, axis=1)[:, :k]
                imputed_values = np.mean(df_numeric.iloc[nearest_neighbors], axis=1)
                df_imputed.loc[missing, col] = imputed_values

        return df_imputed

    def forward_fill(self, df):
        """
        Forward fill missing values.
        """
        return df.ffill()

    def backward_fill(self, df):
        """
        Backward fill missing values.
        """
        return df.bfill()

    def random_imputation(self, df):
        """
        Random imputation (simple random value from the column).
        """
        df_imputed = df.copy()
        for col in df.columns:
            missing = df[col].isna()
            if missing.any():
                random_values = df[col].dropna().sample(missing.sum(), replace=True)
                df_imputed.loc[missing, col] = random_values.values
        return df_imputed


class Processing:
    def __init__(self):
        pass

    def preprocessing(self, dataset: pd.DataFrame, method="drop", strategy="mean", fill_value=None, k=5, subset=None) -> pd.DataFrame:
        """
        Preprocess the input dataset by either dropping missing values or imputing them.
        """
        preprocess = PreProcessing()

        if method == "drop" and isinstance(dataset, pd.DataFrame):
            return preprocess.dropna(dataset, subset=subset)

        elif method == "imputation" and isinstance(dataset, pd.DataFrame):
            return preprocess.imputation(dataset, strategy=strategy, fill_value=fill_value)

        elif method == "knn" and isinstance(dataset, pd.DataFrame):
            return preprocess.knn_imputation(dataset, k=k)

        elif method == "forward_fill" and isinstance(dataset, pd.DataFrame):
            return preprocess.forward_fill(dataset)

        elif method == "backward_fill" and isinstance(dataset, pd.DataFrame):
            return preprocess.backward_fill(dataset)

        elif method == "random" and isinstance(dataset, pd.DataFrame):
            return preprocess.random_imputation(dataset)

        else:
            raise ValueError("Invalid method. Choose from 'drop', 'imputation', 'knn', 'forward_fill', "
                             "'backward_fill', or 'random'.")

    def summary(self, df: pd.DataFrame) -> dict:
        """
        Generate a summary of the input DataFrame, including the number of missing values and constant features.
        """
        constant_features = [col for col in df.columns if df[col].nunique() <= 1]

        return {
            "Missing Values": df.isna().sum(),
            "Constant Features": constant_features
        }

    def remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove all constant features from the input DataFrame.
        """
        constant_features = self.summary(df)["Constant Features"]
        return df.drop(columns=constant_features)

    def remove_outliers(self, df: pd.DataFrame, threshold=1.5) -> pd.DataFrame:
        """
        Remove all outliers from the input DataFrame using the specified threshold.
        """
        qualitatives = df.select_dtypes(include=[object]).columns
        quantitatives = [c for c in df.columns if c not in qualitatives]

        def quantile(s):
            """Identify outliers using quantiles."""
            assert isinstance(s, pd.Series)
            q_025 = s.quantile(.25)
            q_075 = s.quantile(.75)
            iq = q_075 - q_025
            return ~((s > q_075 + threshold * iq) | (s < q_025 - threshold * iq))

        mask = ft.reduce(lambda x, y: x & y, [quantile(df[col]) for col in quantitatives])
        return df.loc[mask].copy()

    def transform_categorical(self, df: pd.DataFrame, l_order: list = None, drop="first") -> dict:
        """
        Transforms categorical variables in the DataFrame to ordinal or one-hot encoded format.
        """
        qualitatives = df.select_dtypes(include=[object]).columns
        ordinal_cols = []
        if l_order:
            if not set(l_order).issubset(set(qualitatives)):
                raise ValueError(f"{l_order} is not included in {list(qualitatives)}")
            ordinal_cols = [c for c in qualitatives if c in l_order]
            ordinal_encoder = PreProcessing()
            ordinals, ordinal_mapping = ordinal_encoder.ordinal_encode(df[ordinal_cols], l_order)
        else:
            ordinals, ordinal_mapping = None, None

        nominal_encoder = PreProcessing()
        nominal_cols = [c for c in qualitatives if c not in ordinal_cols]
        nominals, _ = nominal_encoder.one_hot_encode(df[nominal_cols], drop=drop)  # Ignore the mapping here
        nominal_mapping = None  # No mapping to store

        if ordinal_cols:
            column_names = list(ordinal_cols) + list(nominals.columns)
            return {
                "numericals": pd.DataFrame(
                    data=np.hstack([ordinals, nominals]),
                    columns=column_names),
                "mapping": dict(zip(ordinal_cols, ordinal_mapping)) if ordinal_mapping else None
            }
        else:
            return {"numericals": pd.DataFrame(data=nominals, columns=nominals.columns)}

    def df_to_numerical(self, df: pd.DataFrame, l_order: list = None, drop="first") -> dict:
        """
        Converts the input DataFrame to a numerical representation by transforming its categorical variables.
        """
        numericals_columns = self.transform_categorical(df, l_order, drop)
        qualitatives = df.select_dtypes(include=[object]).columns
        quantitatives = [c for c in df.columns if c not in qualitatives]

        df_transform = numericals_columns["numericals"].reset_index(drop=True) \
            .join(df[quantitatives].reset_index(drop=True), how="inner")

        result = {"df_transform": df_transform}
        if "mapping" in numericals_columns:
            result["ordinal_mapping"] = numericals_columns["mapping"]

        return result

    def replace_missing_values(self, df: pd.DataFrame, missing_values=None) -> pd.DataFrame:
        """
        Replace missing or invalid values with NaN in the DataFrame.
        """
        if missing_values is None:
            missing_values = [None, 0, -1, ".", "-1", "0"]

        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].str.strip().replace(missing_values, np.nan)
            else:
                df[col] = df[col].replace(missing_values, np.nan)

        return df
