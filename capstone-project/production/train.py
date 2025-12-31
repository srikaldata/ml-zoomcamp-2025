#!/usr/bin/env python
# coding: utf-8

# # Estimation of Sales Revenue based on Advertisement Budgets across various channels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# package versions used to verify to create a virtual environment:
# np.__version__
# # '2.3.4'
# pd.__version__
# # '2.3.3'
# matplotlib.__version__
# # '3.10.7'
# sns.__version__
# # '0.13.2'
# sklearn.__version__
# # '1.7.2'
# xgboost.__version__
# # '3.1.1'


# importing sales revenue data
adsales_df = pd.read_csv('data/advertising_and_sales_clean.csv')
print(adsales_df.head())

# info about the df
print(adsales_df.info())

# stats of each col in df
adsales_df.describe(include='all')

# range of sales
min_sales = min(adsales_df['sales'])
max_sales = max(adsales_df['sales'])

print('Range of sales revenue:', min_sales, 'to', max_sales)


# shape of the initial dataframe
print(adsales_df.shape)


# ## 1. Data Cleaning

# checking empty cells
print(adsales_df.isnull().sum())


# checking duplicate records
print('number of duplicate records:', int(adsales_df.duplicated().sum()))


# name of the feature and target columns to check if it is in lowercase
print(list(adsales_df.columns))


# ## 2. Exploratory Data Analysis EDA

# distribution of the target variable --> sales revenue

# plotting the distribution
sns.histplot(adsales_df['sales'], bins=20, kde=True, stat='density', edgecolor='black')


plt.title('Distribution of sales revenue in $')
plt.xlabel('sales revenue $')
plt.ylabel('frequency')
plt.axvline(adsales_df['sales'].mean(), color='red', linestyle='--', label='Mean')
plt.legend()
plt.tight_layout()

# plt.show()


# The target variable 'sales' revenue is almost uniformly distributed

# differentiating categorical and numerical features
# categorical features
cat_features  = list(adsales_df.select_dtypes(include=['object', 'category']).columns)

print('Categorical features:')
print(len(cat_features))
print(cat_features)
print()

# numerical features
num_features = list(adsales_df.select_dtypes(include=np.number).columns)

# removing the target
num_features.remove('sales') 

print('Numerical features:')
print(len(num_features))
print(num_features)

# unique values of categorical variable 'influencer'
print('Unique values of "influencer" feature: ')
print(adsales_df['influencer'].unique())
print(len(adsales_df['influencer'].unique()))


# Influencer categorized tiers explained:
# <br><br>
# Mega	1M+ followers <br>
# Macro	100K–1M followers <br>
# Micro	10K–100K followers <br>
# Nano	1K–10K followers


# distribution of the categorical feature 'influecer'
counts, bins, patches = plt.hist(adsales_df["influencer"], 
                                 bins=len(adsales_df["influencer"].unique()), 
                                 edgecolor='black')

# Count labels on bars
plt.bar_label(patches)

plt.title('Distribution of \'influencer\' column')
plt.xlabel('influencer type')
plt.ylabel('counts')

# Additional text labels (optional, since bar_label already shows counts)
for count, x in zip(counts, bins):
    if count > 0:
        plt.text(x + (bins[1]-bins[0])/2, count, str(int(count)), ha='center', va='bottom')

plt.tight_layout()
# plt.show()



adsales_df['influencer'].value_counts()


# The 'influencer' categorical feature is almost balanced

# distribution of all numerical features

# setting the plots
fig, axes = plt.subplots(3, 1, figsize=(8, 12))
axes = axes.flatten()

# plot each num feature
for i, col in enumerate(num_features):
    # Histogram with density
    axes[i].hist(adsales_df[col], bins=32, density=True, alpha=0.6, edgecolor='black', label='Distribution')

    # Density line (KDE)
    adsales_df[col].plot(kind='density', ax=axes[i], color='red', linewidth=2, label='Density')

    # Mean line
    mean_val = adsales_df[col].mean()
    axes[i].axvline(mean_val, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1f}')

    axes[i].set_title(f'Distribution of \'{col}\' feature')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Density')
    axes[i].legend()

plt.tight_layout()
# plt.show()


# Based on the distributions observed, even though there is a slight skew in 'social_media' and the platykurtic other 2 features, there is no need for any transformations on the give dataset

# correlation matrix of numerical features (between the num features)
corr_matrix = adsales_df[num_features].corr()

# Plotting the heatmap
plt.figure(figsize=(4,3))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation matrix heatmap of tv, radio, social_media budgets')
# plt.show()


# display(corr_matrix)


# pairplot of the numerical features in the advertisement budgets
sns.pairplot(adsales_df[num_features])
# plt.show()


# There is a high collinearity between tv and radio budgets. 
# 
# We can also observe the multicollinearity between all 3 numerical features


# mutual information of features against sales revenue (between features and target)
from sklearn.feature_selection import mutual_info_regression

# data prep for mi calc
mi_features = adsales_df.copy()
mi_features = mi_features.drop(columns='sales')
mi_features['influencer'] = pd.Categorical(mi_features['influencer']).codes  
print()

mi_score = mutual_info_regression(mi_features, adsales_df['sales'], discrete_features=[3], random_state=67)
print('Mutual information scores of each feature against the target variable \'sales\' revenue:')
print(pd.Series(mi_score, index=mi_features.columns).sort_values(ascending=False))

# feature importance of numerical features against sales revenue using pearson correlation
num_feature_corr_w_sales = adsales_df[num_features].corrwith(adsales_df['sales'])

print('Feature importance of numerical features against sales revenue using pearson correlation:')
print(num_feature_corr_w_sales.abs().sort_values(ascending=False))


# **Decision about 'tv' and 'radio' being collinear and also having high mutual info score against sales revenue:**
# <br><br>
# The 'tv' outperforms other advertising channels and second comes 'radio'. 
# 
# But, 'tv' and 'radio' are heavily correlated between them and also has a strong positive correlation with target 'sales' revenue. Strikingly, the visualization in pairplot reveals a thickness of the data points which means a greater spread when radio v/s tv is plotted. Since, radio also remains as a good choice for local advertisements, the feature is decided to be kept.
# 
# Due to the multicollinearity observed, to avoid overfitting, models with regularization parameters are opted to be chosen to fit the data.
# 
# In more advanced scenarios to select features, variance inflation factor VIF or PCA can be used.
# <br><br><br>
# 
# **Decision on keeping or discarding categorical variable 'influencer':**
# <br><br>
# Even though 'influencer' channel seems to be almost independent of predicting 'sales' revenue due to its mutual information score, it is kept and NOT discarded, since it is the only categorical variable available that can increase the complexity of the future operations. 
# 
# In practical cases, it can be discarded and instead values of budgets allocated for each category of influencer channel and their respective budgets can be included as separate features.


# extracting the  the last record for inference and saving it as a csv
# display('Before saving inference record:',adsales_df.tail(3))

adsales_df.tail(1).to_csv('record_for_inference.csv', index=False)

# dropping the inference record
adsales_df = adsales_df.iloc[:-1]

# display('After saving inference record:', adsales_df.tail(3))



# final dataframe and info about the dataframe for analyses
# display(adsales_df.head())
print()
print(len(adsales_df.columns))
print(adsales_df.columns)
print()
print('Numerical features:', len(num_features))
print(num_features)
print()
print('Categorical features:', len(cat_features))
print(cat_features)
print()
print('Total records -->', adsales_df.shape[0])


# Finalized 4 features (3 numerical and 1 categorical) & 1 numerical target variable

# ## 3. Data preparation for modeling

# splitting train, validation, test 60%, 20%, 20% sets
from sklearn.model_selection import train_test_split

full_train_df, test_df = train_test_split(adsales_df, test_size=0.2, random_state=67)

train_df, val_df = train_test_split(full_train_df, test_size=0.25, random_state=67)

# are the splits done right?
print(len(train_df), len(val_df), len(test_df))
print(round(len(train_df)/len(adsales_df), 2), round(len(val_df)/len(adsales_df), 2), round(len(test_df)/len(adsales_df), 2))


# resetting indices of split train val test sets
full_train_df = full_train_df.reset_index(drop=True)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# isolating the target variable sales revenue
y_full_train = full_train_df['sales'].values
y_train = train_df['sales'].values
y_val = val_df['sales'].values
y_test = test_df['sales'].values


# removing target from train val test feature sets
del full_train_df['sales']
del train_df['sales']
del val_df['sales']
del test_df['sales']


# importing dict vectorizer
from sklearn.feature_extraction import DictVectorizer


# fn to vectorize dataframes
def vectorize_dfs(train_or_full_df, val_or_test_df):

    # instantiate the dictionary vectorizer
    dictionary_vectorizer = DictVectorizer(sparse=False)

    # create x train or full train after converting the features to dictionary and vectorizing features
    X_train_or_full = dictionary_vectorizer.fit_transform(train_or_full_df[num_features + cat_features].to_dict(orient='records'))

    # transform the val or test set features by converting them to py dict and applying the transformation
    X_val_or_test = dictionary_vectorizer.transform(val_or_test_df[num_features + cat_features].to_dict(orient='records'))

    # extract feature names
    feature_names = dictionary_vectorizer.get_feature_names_out()

    print('Vectorizing successful!')

    return X_train_or_full, X_val_or_test, feature_names


# vectorizing features of train and validation 
X_train, X_val, feature_names_train = vectorize_dfs(train_df, val_df)

# vectorizing features of full train and test 
X_full_train, X_test, feature_names_full_train = vectorize_dfs(full_train_df, test_df)


# checking the first vectors of train, val, test feature sets 
print(X_train[:1])
print()
print(X_val[:1])
print()
print()
print(X_full_train[:1])
print()
print(X_test[:1])


# ## 4. Training the models

# importing the needed regressor model frameworks 
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error



# constructing pipelines for each model
pipelines = {
    'ridge': Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(random_state=67))
    ]),
    'elastic_net': Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(random_state=67))
    ]),
    'random_forest': Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=67))
    ]),
    'gradient_boosting': Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(random_state=67))
    ])
}


# constructing hyperparameter grids for tuning
param_grids = {
    'ridge': {
        'model__alpha': [0.1, 1.0, 10.0, 100.0]
    },
    'elastic_net': {
        'model__alpha': [0.1, 1.0, 10.0],
        'model__l1_ratio': [0.2, 0.5, 0.8]
    },
    'random_forest': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5]
    },
    'gradient_boosting': {
        'model__n_estimators': [100, 200],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__max_depth': [3, 4, 5]
    }
}



# 5 fold KFold for regression
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=67)

# custom fn to train and tune models using the training data
def train_and_tune_regression(pipelines, param_grids, X_train_or_full, y_train_or_full):
    """Train regression models with GridSearchCV and return the best estimators."""

    best_estimators = {}

    for name in pipelines:
        print(f"Training and tuning {name}...")

        # grid search hyperparameter tuning using negative mean sq error making -mse the scoring and deciding metric  
        grid = GridSearchCV(estimator=pipelines[name], param_grid=param_grids[name], cv=cv_splitter, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

        # fitting the data
        grid.fit(X_train_or_full, y_train_or_full)

        # extracting the best params
        print(f"Best params for {name}: {grid.best_params_}")

        # extracting the best estimator
        best_estimators[name] = grid.best_estimator_
        print("-" * 75)

    return best_estimators


best_reg_model_params = train_and_tune_regression(pipelines, param_grids, X_train, y_train)



# the complete model parameters of regularized and tuned models
print(best_reg_model_params)


# ## 5. Predictions and evaluations


# custom fn to predict and evaluate and plot the eval metrics
def predict_and_evaluate_regression(best_model_reg_params, X_val_or_test, y_val_or_test):
    """Use best estimators to predict and print regression evaluation metrics."""

    # EVALUATION:

    # empty dict of lists to store metrics 
    scores = {name: [] for name in best_model_reg_params.keys()}

    # for adjusted R2
    n = len(y_val_or_test)
    p = X_val_or_test.shape[1]

    for name, model in best_model_reg_params.items():
        print(f"Evaluating model {name}...")
        y_pred = model.predict(X_val_or_test)

        # calculate eval metrics
        r2 = r2_score(y_val_or_test, y_pred)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        rmse = root_mean_squared_error(y_val_or_test, y_pred)
        mae = mean_absolute_error(y_val_or_test, y_pred)

        # storing the scores
        scores[name] = [r2, adj_r2, rmse, mae]

        print(f"R2 Score: {r2:.4f}")
        print(f"Adj R2: {adj_r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print("-" * 50)

    # VISUALIZATION:
    model_names = list(scores.keys())
    metric_values = np.array(list(scores.values()))  # shape: [n_models, 4]
    indices = np.arange(len(model_names))
    bar_width = 0.5

    # Create subplots (2x2 grid)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Regression Model Comparison Across Metrics', fontsize=14)

    metric_titles = ['R² Score', 'Adjusted R²', 'RMSE', 'MAE']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    for i, ax in enumerate(axs.flat):
        ax.bar(indices, metric_values[:, i], color=colors[i], width=bar_width)
        ax.set_title(metric_titles[i])
        ax.set_xticks(indices)
        ax.set_xticklabels(model_names, rotation=30, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # annotate bar labels
        for idx, val in enumerate(metric_values[:, i]):
            ax.text(idx, val + 0.01 * np.max(metric_values[:, i]), f'{val:.3f}',
                    ha='center', va='bottom', fontsize=8)

        # adjust y-label based on metric type
        if i in [0, 1]:
            ax.set_ylabel('Score')
        else:
            ax.set_ylabel('Error')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.show()

    return scores


# making the prediction and retrieving the scores on validation dataset
val_set_scores = predict_and_evaluate_regression(best_reg_model_params, X_val, y_val)


# ### Final model rankings (production ready)
# 
# | Rank | Regression Model          | RMSE   | MAE    | R²     | Verdict                  |
# |------|----------------|--------|--------|--------|--------------------------|
# | 🥇   | **Ridge**      | **2,903** | **2,309** | **0.9990** | **Simplest + Interpretable** |
# | 🥈   | **Gradient Boost**         | 2,982 | 2,362 | 0.9990 | Second best + robust     |
# | 🥉   | **Random Forest**         | 3,221  | 2,557  | 0.9988    | Solid tree ensemble      |
# | 4th  | **Elastic Net**    | 4,363  | 3,500  | 0.9978    | Still good, but beaten  |

# **Reasons for selecting Ridge regression model for production:**
# 1. the Ridge regression outperforms other models across all metrics 
# 2. the coefficients of Ridge are much simpler to interpret
# 3. the model itself is lightweight and can be hosted efficiently

# ## 6. Best model and its feature importances 


# The best model --> ridge regression with the following params
# display(best_reg_model_params['ridge'].named_steps['model'])


# instantiating the best model
ridge_final = best_reg_model_params['ridge']

# fitting all the training data
ridge_final.fit(X_full_train, y_full_train)

# make the predictions on the holdout test set
y_pred = ridge_final.predict(X_test)

# printing the evaluation metrics
print('Evaluation metrics on holdout test set:')
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")


# The model performs really well. It performs better with test set in few metrics and almost similar in other metrics when compared to evaluation on validation set 


# feature importances using the coefficients of 
ridge_coefs = ridge_final.named_steps['model'].coef_
ridge_intercept = ridge_final.named_steps['model'].intercept_

ridge_coefs_df = pd.DataFrame({
    'Feature': feature_names_full_train, 
    'Coefficient': ridge_coefs,
    'Sign': ['Positive' if coef > 0 else 'Negative' for coef in ridge_coefs]
})
ridge_coefs_df = ridge_coefs_df.sort_values(by='Coefficient', ascending=False)
ridge_coefs_df = pd.concat([pd.DataFrame({'Feature': ['INTERCEPT'],'Coefficient': [ridge_intercept], 'Sign': ['INTERCEPT']}), ridge_coefs_df])

# display(ridge_coefs_df)



# feature importances from coefficients 
plt.figure(figsize=(12, 4))
bars = plt.bar(ridge_coefs_df['Feature'][2:], ridge_coefs_df['Coefficient'][2:], color='skyblue')
plt.title('Feature importances from coefficients ranked (Except \'tv\' which has coef of 93026.384) ')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}', ha='center', va='baseline')

plt.tight_layout()
# plt.show()


# ## 7. Pickling the best regression model for inference


# saving the model as a pickled file
import pickle

with open('pickled_model.bin', 'wb') as file:
    pickle.dump(ridge_final, file)

print('successfully exported the best model!')

# instantiate dict vectorizer for inference
dictvectorizer_fulltrain = DictVectorizer(sparse=False)

# create x train or full train after vectorizing features respectively
X_train_full = dictvectorizer_fulltrain.fit_transform(full_train_df.iloc[:, :10].to_dict(orient='records'))

# exporting the dict vectorizer
with open('dictvectorizer_fulltrain.bin', 'wb') as file:
    pickle.dump(dictvectorizer_fulltrain, file)

print('successfully exported the fitted dict vectorizer!')

# testing the pickled model
with open('pickled_model.bin', 'rb') as file:
    loaded_reg_model = pickle.load(file)

# loaded_model to make predictions
y_pred = loaded_reg_model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_test, y_pred):.4f}")

print('model imported and the test is successful!')

# testing the exported final dict vectorizer 
with open('dictvectorizer_fulltrain.bin', 'rb') as file:
    loaded_dictvectorizer = pickle.load(file)

# transforming the feature of test data as a test 
feature_inference = loaded_dictvectorizer.transform(test_df.iloc[:, :-1].to_dict(orient='records'))
feature_inference


