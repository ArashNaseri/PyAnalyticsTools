from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


def decision_tree_imputation(df: pd.DataFrame, thresh: float = 0.5) -> pd.DataFrame:
    """
    Perform imputation of missing values in a DataFrame using a decision tree regressor.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to perform imputation on.
    thresh : float, optional (default=0.5)
        The threshold for correlation between columns. If the absolute correlation
        between a column with missing values and other numeric columns in the
        DataFrame is below this threshold, then the missing values will be
        imputed using the median of the column. Otherwise, the missing values
        will be imputed using a decision tree regressor trained on the columns
        with high correlation to the missing column.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with missing values imputed using a decision
        tree regressor.

    Raises
    ------
    ValueError
        If the input DataFrame contains non-numeric data, which cannot be
        handled by the decision tree regressor.
    """
    df_imputed = df.copy()
    for col in df_imputed.columns[df_imputed.isna().sum() > 0]:
        if df_imputed[col].dtype == np.number:
            # Select only features with high correlation to the current column
            corr = df_imputed.corr()[col]
            features = corr[(corr.abs() > thresh) & (corr.index != col)].index.tolist()
            if len(features) == 0:
                # If no features have high enough correlation, impute with median
                imp = SimpleImputer(strategy='median')
                df_imputed[col] = imp.fit_transform(df_imputed[[col]])
                print(f"{col}: imputed with median")
            else:
                # Impute with decision tree regression
                X = df_imputed[features]
                y = df_imputed[col]
                imp = SimpleImputer(strategy='mean')
                X_imputed = imp.fit_transform(X)
                reg = DecisionTreeRegressor().fit(X_imputed, y)
                X_missing = X[df_imputed[col].isna()]
                X_missing_imputed = imp.transform(X_missing)
                y_pred = reg.predict(X_missing_imputed)
                df_imputed.loc[df_imputed[col].isna(), col] = y_pred
                print(f"{col}: imputed with decision tree regression using {', '.join(features)}")
        else:
            # If column is not numeric, impute with mode
            imp = SimpleImputer(strategy='most_frequent')
            df_imputed[col] = imp.fit_transform(df_imputed[[col]])
            print(f"{col}: imputed with mode")
    return df_imputed
