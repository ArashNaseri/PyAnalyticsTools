# PyAnalyticsTools


The repository is a collection of useful tools and functions developed by the author for data analysis and machine learning tasks. These tools are designed to be easy to use and adaptable to different types of data, making them ideal for a wide range of applications. The repository is regularly updated with new tools and functions developed by the author based on their ongoing projects and research. Users can explore the repository to find tools that are relevant to their work and can easily integrate them into their own projects.


**decision_tree_imputation** :

  _decision_tree_imputationis_ a Python function that performs imputation of missing values in a Pandas DataFrame using a decision tree regressor. Missing data is a common problem in data science and machine learning projects, and imputation is a technique used to fill in missing data with estimated values. The function first identifies the columns in the input DataFrame that contain missing values and checks if they are numeric. If a column is numeric, the function selects the other numeric columns in the DataFrame that have a high correlation with the column with missing values. If no columns have a high correlation, the function imputes the missing values with the median of the column. Otherwise, the function trains a decision tree regressor on the highly correlated columns and uses it to predict the missing values. If a column is not numeric, the function imputes the missing values with the mode of the column.

  The function takes two parameters: df and thresh. df is the Pandas DataFrame that needs imputation, and thresh is the threshold for correlation between columns. If the absolute correlation between a column with missing values and other numeric columns in the DataFrame is below this threshold, then the missing values will be imputed using the median of the column. Otherwise, the missing values will be imputed using a decision tree regressor trained on the columns with high correlation to the missing column.

  The function returns a copy of the input DataFrame with missing values imputed using a decision tree regressor. If the input DataFrame contains non-numeric data, which cannot be handled by the decision tree regressor, the function raises a ValueError exception.

  This function is designed to be a useful tool for data cleaning and preprocessing in data science and machine learning projects. It can save time and effort in dealing with missing data and can help improve the quality of the resulting models. It is also designed to be flexible and customizable, allowing users to adjust the correlation threshold and easily modify the imputation strategy for non-numeric data.
