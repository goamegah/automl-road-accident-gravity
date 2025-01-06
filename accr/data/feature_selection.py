import pandas as pd
import numpy as np
import functools as ft
from scipy.stats import chi2_contingency, f_oneway

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


class Processing:
    def __init__(self):
        self.preprocessing_utils = PreProcessing()

    def detect_variable_types(self, df: pd.DataFrame) -> dict:
        """
        Detect and separate numerical and categorical variables in the DataFrame.
        """
        categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        return {"categorical": categorical_vars, "numerical": numerical_vars}

    def encode_categorical(self, df: pd.DataFrame, ordinal_cols: dict = None, drop="first") -> pd.DataFrame:
        """
        Encode categorical variables using one-hot or ordinal encoding.
        """
        detected_types = self.detect_variable_types(df)
        categorical_vars = detected_types["categorical"]

        # Ordinal encoding
        if ordinal_cols:
            for col, order in ordinal_cols.items():
                if col in categorical_vars:
                    df[col], _ = self.preprocessing_utils.ordinal_encode(df[[col]], order)
                    categorical_vars.remove(col)

        # One-hot encoding for remaining categorical variables
        if categorical_vars:
            df = pd.get_dummies(df, columns=categorical_vars, drop_first=(drop == "first"))
        
        return df

    def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing values for numerical and categorical variables.
        """
        detected_types = self.detect_variable_types(df)
        for col in detected_types["numerical"]:
            df[col] = self.preprocessing_utils.imputation(df[[col]], strategy="mean")
        for col in detected_types["categorical"]:
            df[col] = self.preprocessing_utils.imputation(df[[col]], strategy="mode")
        return df

    def remove_outliers(self, df: pd.DataFrame, threshold=1.5) -> pd.DataFrame:
        """
        Remove all outliers from the input DataFrame using the specified threshold (IQR method).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        def quantile_mask(s):
            """Create a mask for outliers using IQR."""
            Q1 = s.quantile(0.25)
            Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (s >= lower_bound) & (s <= upper_bound)

        # Combine masks for all numeric columns using functools.reduce
        combined_mask = ft.reduce(
            lambda acc, col: acc & quantile_mask(df[col]),
            numeric_cols,
            pd.Series(True, index=df.index)
        )
        
        # Return the DataFrame filtered by the combined mask
        return df[combined_mask].copy()

    def select_features_corr(self, df, target_column, threshold=0.1, problem_type="classification"):
        """
        Select features for a DataFrame containing mixed data (numerical + categorical) without Scikit-learn.
        
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            target_column (str): Target column.
            threshold (float): Threshold for feature selection (correlation or association).
            problem_type (str): Type of problem ('classification' or 'regression').
        
        Returns:
            pd.DataFrame: DataFrame with selected features.
            dict: Importance scores for the features.
        """
        if target_column not in df.columns:
            raise ValueError(f"The target column '{target_column}' does not exist in the DataFrame.")

        # Step 1: Identify variable types
        detected_types = self.detect_variable_types(df)
        numerical_cols = detected_types["numerical"]
        categorical_cols = detected_types["categorical"]

        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

        # Step 2: Compute scores for numerical variables
        scores = {}
        selected_numerical = []
        if numerical_cols:
            if problem_type == "classification":
                # Pearson correlation for classification
                correlation_matrix = df[numerical_cols + [target_column]].corr()
                target_corr = correlation_matrix[target_column].drop(target_column)
                selected_numerical = target_corr[abs(target_corr) >= threshold].index.tolist()
                scores.update(target_corr.to_dict())
            elif problem_type == "regression":
                # ANOVA F-test for regression
                numerical_scores = {}
                for col in numerical_cols:
                    f_value, _ = f_oneway(df[col], df[target_column])
                    numerical_scores[col] = f_value
                selected_numerical = [col for col, score in numerical_scores.items() if score >= threshold]
                scores.update(numerical_scores)

        # Step 3: Compute scores for categorical variables
        selected_categorical = []
        if categorical_cols:
            categorical_scores = {}
            for col in categorical_cols:
                if problem_type == "classification":
                    # Chi-squared test
                    contingency_table = pd.crosstab(df[col], df[target_column])
                    chi2, _, _, _ = chi2_contingency(contingency_table)
                    categorical_scores[col] = chi2
                elif problem_type == "regression":
                    # ANOVA F-test (ordinal encoding for categories)
                    ordinal_mapping = {val: idx for idx, val in enumerate(df[col].dropna().unique())}
                    ordinal_values = df[col].map(ordinal_mapping)
                    f_value, _ = f_oneway(ordinal_values, df[target_column])
                    categorical_scores[col] = f_value

            # Filter categorical variables based on the threshold
            selected_categorical = [col for col, score in categorical_scores.items() if score >= threshold]
            scores.update(categorical_scores)

        # Step 4: Combine selected variables
        selected_features = selected_numerical + selected_categorical

        # Return DataFrame with selected features
        return df[selected_features + [target_column]], scores

    def pca_reduction(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """
        Reduce dimensionality using Principal Component Analysis (PCA).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        data = df[numeric_cols]
        data_centered = data - data.mean()
        cov_matrix = np.cov(data_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
        reduced_data = np.dot(data_centered, eigenvectors)
        return pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])

    def pca_for_mixed_data(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """
        Apply PCA to mixed data by encoding categorical variables and standardizing numerical ones.
        """
        df_encoded = self.encode_categorical(df)  # Encode categorical variables
        df_standardized = self.standardize(df_encoded)  # Standardize data
        return self.pca_reduction(df_standardized, n_components)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize numerical columns using Min-Max scaling.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
        return df

    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize numerical columns using Z-score normalization.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
        return df
    
    def replace_missing_values(self, df: pd.DataFrame, missing_values=None) -> pd.DataFrame:
        """
        Replace missing or invalid values with NaN in the DataFrame.
        """
        if missing_values is None:
            missing_values = [None, 0, -1, ".", "-1", "0", "NA", "N/A", "nan", "#VALEURMULTI", "#ERREUR"]

        for col in df.columns:
            if df[col].dtype in ['object', 'category']:
                df[col] = df[col].str.strip().replace(missing_values, np.nan)
            else:
                df[col] = df[col].replace(missing_values, np.nan)

        return df

    def summary(self, df: pd.DataFrame):
        """
        Provide a summary of the DataFrame, including numerical and categorical columns,
        missing values, and other key details.
        """
        detected_types = self.detect_variable_types(df)
        summary = {
            'Total Columns': df.shape[1],
            'Total Rows': df.shape[0],
            'Numerical Columns': detected_types['numerical'],
            'Categorical Columns': detected_types['categorical'],
            'Missing Values': df.isnull().sum().to_dict()
        }
        return summary
    
    def drop_columns_with_missing(self, df, threshold_ratio=0.5):
        """
        Supprime les colonnes ayant un trop grand nombre de valeurs manquantes.
        
        Args:
            summary (dict): Résumé des données (output de la méthode summary).
            df (pd.DataFrame): DataFrame original.
            threshold_ratio (float): Seuil en proportion (e.g., 0.5 pour 50% de valeurs manquantes).
        
        Returns:
            pd.DataFrame: DataFrame sans les colonnes inutilisables.
            list: Liste des colonnes supprimées.
        """
        summary = self.summary(df)
        total_rows = summary['Total Rows']
        missing_values = summary['Missing Values']
        
        # Seuil basé sur le ratio
        threshold = total_rows * threshold_ratio
        
        # Identifier les colonnes à supprimer
        columns_to_drop = [col for col, missing in missing_values.items() if missing > threshold]
        
        # Supprimer les colonnes
        df_cleaned = df.drop(columns=columns_to_drop)
        
        return df_cleaned, columns_to_drop





# import pandas as pd
# import numpy as np
# import functools as ft
# from scipy.stats import chi2_contingency, f_oneway


# class PreProcessing:
#     def ordinal_encode(self, df, l_order):
#         """
#         Custom implementation of ordinal encoding.
#         """
#         mapping = {col: {val: idx for idx, val in enumerate(l_order)} for col in df.columns}
#         encoded_df = df.apply(lambda col: col.map(mapping[col.name]) if col.name in mapping else col)
#         return encoded_df, mapping

#     def one_hot_encode(self, df, drop="first"):
#         """
#         Custom one-hot encoding.
#         """
#         encoded_df = pd.get_dummies(df, drop_first=(drop == "first"))
#         return encoded_df, None  # Return None for mapping, as it's not required for one-hot encoding

#     def dropna(self, df, subset=None):
#         """
#         Custom dropna method.
#         """
#         return df.dropna(subset=subset)

#     def imputation(self, df, strategy="mean", fill_value=None):
#         """
#         Impute missing values based on the strategy.
#         """
#         if strategy == "mean":
#             return df.fillna(df.mean())
#         elif strategy == "median":
#             return df.fillna(df.median())
#         elif strategy == "mode":
#             return df.fillna(df.mode().iloc[0])
#         elif fill_value is not None:
#             return df.fillna(fill_value)
#         else:
#             raise ValueError("Invalid strategy or fill_value.")

#     def forward_fill(self, df):
#         """
#         Forward fill missing values.
#         """
#         return df.ffill()

#     def backward_fill(self, df):
#         """
#         Backward fill missing values.
#         """
#         return df.bfill()


# class Processing:
#     def __init__(self):
#         self.preprocessing_utils = PreProcessing()

#     def detect_variable_types(self, df: pd.DataFrame) -> dict:
#         """
#         Detect and separate numerical and categorical variables in the DataFrame.
#         """
#         categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
#         numerical_vars = df.select_dtypes(include=[np.number]).columns.tolist()
#         return {"categorical": categorical_vars, "numerical": numerical_vars}

#     def encode_categorical(self, df: pd.DataFrame, ordinal_cols: dict = None, drop="first") -> pd.DataFrame:
#         """
#         Encode categorical variables using one-hot or ordinal encoding.
#         """
#         detected_types = self.detect_variable_types(df)
#         categorical_vars = detected_types["categorical"]

#         # Ordinal encoding
#         if ordinal_cols:
#             for col, order in ordinal_cols.items():
#                 if col in categorical_vars:
#                     df[col], _ = self.preprocessing_utils.ordinal_encode(df[[col]], order)
#                     categorical_vars.remove(col)

#         # One-hot encoding for remaining categorical variables
#         if categorical_vars:
#             df = pd.get_dummies(df, columns=categorical_vars, drop_first=(drop == "first"))
        
#         return df

#     def impute_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Impute missing values for numerical and categorical variables.
#         """
#         detected_types = self.detect_variable_types(df)
#         for col in detected_types["numerical"]:
#             df[col] = self.preprocessing_utils.imputation(df[[col]], strategy="mean")
#         for col in detected_types["categorical"]:
#             df[col] = self.preprocessing_utils.imputation(df[[col]], strategy="mode")
#         return df

#     def remove_outliers(self, df: pd.DataFrame, threshold=1.5) -> pd.DataFrame:
#         """
#         Remove all outliers from the input DataFrame using the specified threshold (IQR method).
#         """
#         numeric_cols = df.select_dtypes(include=[np.number]).columns

#         def quantile_mask(s):
#             """Create a mask for outliers using IQR."""
#             Q1 = s.quantile(0.25)
#             Q3 = s.quantile(0.75)
#             IQR = Q3 - Q1
#             lower_bound = Q1 - threshold * IQR
#             upper_bound = Q3 + threshold * IQR
#             return (s >= lower_bound) & (s <= upper_bound)

#         # Combine masks for all numeric columns using functools.reduce
#         combined_mask = ft.reduce(
#             lambda acc, col: acc & quantile_mask(df[col]),
#             numeric_cols,
#             pd.Series(True, index=df.index)
#         )
        
#         # Return the DataFrame filtered by the combined mask
#         return df[combined_mask].copy()

#     def select_features_corr(self, df, target_column, threshold=0.1, problem_type="classification"):
#         """
#         Select features for a DataFrame containing mixed data (numerical + categorical) without Scikit-learn.
        
#         Args:
#             df (pd.DataFrame): DataFrame containing the data.
#             target_column (str): Target column.
#             threshold (float): Threshold for feature selection (correlation or association).
#             problem_type (str): Type of problem ('classification' or 'regression').
        
#         Returns:
#             pd.DataFrame: DataFrame with selected features.
#             dict: Importance scores for the features.
#         """
#         if target_column not in df.columns:
#             raise ValueError(f"The target column '{target_column}' does not exist in the DataFrame.")

#         # Step 1: Identify variable types
#         detected_types = self.detect_variable_types(df)
#         numerical_cols = detected_types["numerical"]
#         categorical_cols = detected_types["categorical"]

#         if target_column in numerical_cols:
#             numerical_cols.remove(target_column)
#         if target_column in categorical_cols:
#             categorical_cols.remove(target_column)

#         # Step 2: Compute scores for numerical variables
#         scores = {}
#         selected_numerical = []
#         if numerical_cols:
#             if problem_type == "classification":
#                 # Pearson correlation for classification
#                 correlation_matrix = df[numerical_cols + [target_column]].corr()
#                 target_corr = correlation_matrix[target_column].drop(target_column)
#                 selected_numerical = target_corr[abs(target_corr) >= threshold].index.tolist()
#                 scores.update(target_corr.to_dict())
#             elif problem_type == "regression":
#                 # ANOVA F-test for regression
#                 numerical_scores = {}
#                 for col in numerical_cols:
#                     f_value, _ = f_oneway(df[col], df[target_column])
#                     numerical_scores[col] = f_value
#                 selected_numerical = [col for col, score in numerical_scores.items() if score >= threshold]
#                 scores.update(numerical_scores)

#         # Step 3: Compute scores for categorical variables
#         selected_categorical = []
#         if categorical_cols:
#             categorical_scores = {}
#             for col in categorical_cols:
#                 if problem_type == "classification":
#                     # Chi-squared test
#                     contingency_table = pd.crosstab(df[col], df[target_column])
#                     chi2, _, _, _ = chi2_contingency(contingency_table)
#                     categorical_scores[col] = chi2
#                 elif problem_type == "regression":
#                     # ANOVA F-test (ordinal encoding for categories)
#                     ordinal_mapping = {val: idx for idx, val in enumerate(df[col].dropna().unique())}
#                     ordinal_values = df[col].map(ordinal_mapping)
#                     f_value, _ = f_oneway(ordinal_values, df[target_column])
#                     categorical_scores[col] = f_value

#             # Filter categorical variables based on the threshold
#             selected_categorical = [col for col, score in categorical_scores.items() if score >= threshold]
#             scores.update(categorical_scores)

#         # Step 4: Combine selected variables
#         selected_features = selected_numerical + selected_categorical

#         # Return DataFrame with selected features
#         return df[selected_features + [target_column]], scores

#         # ...

#     def pca_reduction(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
#         """
#         Reduce dimensionality using Principal Component Analysis (PCA).
#         """
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         data = df[numeric_cols]
#         data_centered = data - data.mean()
#         cov_matrix = np.cov(data_centered.T)
#         eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
#         sorted_indices = np.argsort(eigenvalues)[::-1]
#         eigenvectors = eigenvectors[:, sorted_indices[:n_components]]
#         reduced_data = np.dot(data_centered, eigenvectors)
#         return pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])

#     def pca_for_mixed_data(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
#         """
#         Apply PCA to mixed data by encoding categorical variables and standardizing numerical ones.
#         """
#         df_encoded = self.encode_categorical(df)  # Encode categorical variables
#         df_standardized = self.standardize(df_encoded)  # Standardize data
#         return self.pca_reduction(df_standardized, n_components)

#     def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Normalize numerical columns using Min-Max scaling.
#         """
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].min()) / (df[numeric_cols].max() - df[numeric_cols].min())
#         return df

#     def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
#         """
#         Standardize numerical columns using Z-score normalization.
#         """
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
#         return df
    
#     def replace_missing_values(self, df: pd.DataFrame, missing_values=None) -> pd.DataFrame:
#         """
#         Replace missing or invalid values with NaN in the DataFrame.
#         """
#         if missing_values is None:
#             missing_values = [None, 0, -1, ".", "-1", "0", "NA", "N/A", "nan", "#VALEURMULTI"]

#         for col in df.columns:
#             if df[col].dtype in ['object', 'category']:
#                 df[col] = df[col].str.strip().replace(missing_values, np.nan)
#             else:
#                 df[col] = df[col].replace(missing_values, np.nan)

#         return df
