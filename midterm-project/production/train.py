#!/usr/bin/env python
# coding: utf-8

# # Multi-Class Prediction of Insulin Dosage Recommendation Evaluated from Patient Health Data

# IMPORTANT_NOTE: if you are running this file as a script make sure the following is considered:
# 1. the data folder containing the raw patient health data is present in the same directory as this script
# 2. make sure to close the plots generated for the script to proceed to the next line of execution
# 3. if you want to make the verification process easy, the modeling.ipynb file, up the current production directory contains the same code in a notebook format

# importng the needed packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Versions of the packages used :
# verified to create a local env
# np.__version__
# pd.__version__
# matplotlib.__version__
# sns.__version__
# sklearn.__version__
# xgboost.__version__



# importing patient health data

pathealth_df = pd.read_csv('data/insulin_dosage_prediction.csv')

pathealth_df


# overview of the df
pathealth_df.info()

# basic stats of the patient health df
pathealth_df.describe(include='all')


# # 1. Data Cleaning, EDA & Feature selection

# converting column inames into lowercase for easy reference
# since all columns are string types
pathealth_df.columns = pathealth_df.columns.str.lower()

pathealth_df.head(1)


# checking null values
pathealth_df.isnull().sum()


# checking for duplicates
print('num of duplicate records:', int(pathealth_df.duplicated().sum()))



# distribution of the target variable

# plotting the distribution
counts, bins, patches = plt.hist(pathealth_df['insulin'], bins=len(pathealth_df['insulin'].unique()), edgecolor='black')

# plot the labels
bin_width = bins[1] - bins[0]

for count, bin_edge in zip(counts, bins):
    if count > 0:
        plt.text(bin_edge + bin_width/2, count-500, str(int(count)), ha='center', va='bottom')
        plt.text(bin_edge + bin_width/2, count, str(count/len(pathealth_df['insulin'])), ha='center', va='bottom')


plt.title('insulin')
plt.xlabel('insulin decisions')
plt.ylabel('Count')

plt.show()



pathealth_df['insulin'].value_counts()


# The dataset is heavily imbalanced based on the target variable "insulin"
# 
# The scope of this project is only to identify if insulin needs to be increased (up) or kept the same (steady)
# 
# As per the scope, we can drop the records with other categories


# dropping unwanted target variables
pathealth_df.drop(pathealth_df[pathealth_df['insulin'].isin(['no', 'down'])].index, inplace=True)


# checking the modified dataframe
print(pathealth_df.shape)


# distribution of the target variable after modifying the dataframe

# plotting the distribution
counts, bins, patches = plt.hist(pathealth_df['insulin'], bins=len(pathealth_df['insulin'].unique()), edgecolor='black')

# plot the labels
bin_width = bins[1] - bins[0]

for count, bin_edge in zip(counts, bins):
    if count > 0:
        plt.text(bin_edge + bin_width/2, count-500, str(int(count)), ha='center', va='bottom')
        plt.text(bin_edge + bin_width/2, count, str(round(count/len(pathealth_df['insulin'])*100,3))+'%', ha='center', va='bottom')


plt.title('insulin')
plt.xlabel('insulin decisions')
plt.ylabel('Count')

plt.show()


# does the patient id recur and if it matters?
int((pathealth_df['patient_id'].value_counts() > 2).sum())



# the patient id does not recur, so there is no need to group finding for each patient
# each record is assumed to be from individual patient 
# so in no way the patient id is useful
# drop the patient id column as it is an un important feature

pathealth_df.drop('patient_id', axis=1, inplace=True)


# separating the column based on datatypes
# categorical features
cat_features  = list(pathealth_df.select_dtypes(include=['object', 'category']).columns)

# remove the target var
cat_features.remove('insulin') 

print(len(cat_features))
print(cat_features)
print()

# numerical features
num_features = list(pathealth_df.select_dtypes(include=np.number).columns)

print(len(num_features))
print(num_features)



# finding the unique values of categorical variables
for col in cat_features:
    print(col, ':', pathealth_df[col].unique())
    print()



# plotting the distributions of the categorical features

# subplot grid
fig, axes = plt.subplots(1, 4, figsize=(20, 4))

# flatten the axes
axes = axes.flatten()

# plot each cat feature
for i, col in enumerate(cat_features):
    
    # plotting
    counts, bins, patches = axes[i].hist(pathealth_df[col], bins=len(pathealth_df[col].unique()), edgecolor='black')
    
    # count labels
    axes[i].bar_label(patches)
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    
    # plotting the labels
    for count, x in zip(counts, bins):
        if count > 0:
            axes[i].text(x + (bins[1]-bins[0])/2, count, str(int(count)), 
                         ha='center', va='bottom')

# space between subplots
plt.subplots_adjust(wspace=0.2, hspace=0.4)
plt.show()


# value counts of each categorical features
for feature in cat_features:
    print(pathealth_df[feature].value_counts())
    print()


# Almost all categorical features have approximately even representation across its categories



# mutual information of categorical features with insulin
from sklearn.metrics import mutual_info_score

# udf creation
def mutual_info_score_insulin(series):
    return mutual_info_score(series, pathealth_df['insulin'])

# calculating mi for all cat features and sorting by most important at the top
mi_insulin = pathealth_df[cat_features].apply(mutual_info_score_insulin)
mi_insulin.sort_values(ascending=False)



# # testing the above after converting into numerical encoded data
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import mutual_info_score

# ip_df = pathealth_df.copy()
# op_df = pd.DataFrame()

# # Encode categorical features and target
# label_encoders = {}
# for col in cat_features:
#     le = LabelEncoder()
#     op_df[col] = le.fit_transform(ip_df[col])

# # Separate features and target
# target_series = pathealth_df['insulin'].map({'steady': int(0), 'up': int(1)}).reset_index(drop=True)

# # udf creation
# def mutual_info_score_insulin(series):
#     return mutual_info_score(series, target_series)

# # calculating mi for all cat features and sorting by most important at the top
# mi_insulin_series = op_df[cat_features].apply(mutual_info_score_insulin)
# mi_insulin_series.sort_values(ascending=False)




# due to the independence of 'gender' and 'food_intake' with the target variable, dropping those column since based on their mutual info score 





# dropping the cat features which are independent
pathealth_df.drop('gender', axis=1, inplace=True)
cat_features.remove('gender')

pathealth_df.drop('food_intake', axis=1, inplace=True)
cat_features.remove('food_intake')



print(pathealth_df.head())
print()
print(pathealth_df.columns)
print()
print(cat_features)


# distributions of all numerical features

# subplot grid
fig, axes = plt.subplots(3, 3, figsize=(14, 12))

# flatten the axes
axes = axes.flatten()

# plot each num feature
for i, col in enumerate(num_features):
    
    # plotting
    counts, bins, patches = axes[i].hist(pathealth_df[col], bins=32, edgecolor='black')

    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')

# space between subplots
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.show()


# Most of the numerical features are approximately uniformly distributed




# correlation matrix of numerical features
corr_matrix = pathealth_df[num_features].corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap of Patient Health data')
plt.show()



# pairplot of the numerical features in the patient health data
sns.pairplot(pathealth_df)
plt.show()



# target variable numerical encoding conversion
# 'steady': 0
# 'up': 1

pathealth_df['insulin'] = pathealth_df['insulin'].map({'steady': 0, 'up': 1})

pathealth_df['insulin'].value_counts()



# feature importance estimation using pearson correlation
num_feature_corr_w_insulin = pathealth_df[num_features].corrwith(pathealth_df['insulin'])

num_feature_corr_w_insulin.abs().sort_values(ascending=False)


# From the feature-to-feature correlation plots: [correlation matrix (math check) and pairplot (visual check)], 
# 
# Multi collinearity can be confirmed to be absent between the numerical features
# <br>
# <br>
# <br>
# From the feature-to-target correlation calculations:
# 
# 'age' is non-correlated (pearson) feature. While it may be non-linearly correlated, but for the scope of this project we can drop the feature
# 


# dropping age
pathealth_df.drop('age', axis=1, inplace=True)
num_features.remove('age')

pathealth_df.reset_index(drop=True)



# extracting the  the last record for inference and saving it as a csv
print('Before saving inference record:',pathealth_df.tail(3))

pathealth_df.tail(1).to_csv('record_for_inference.csv', index=False)

# dropping the inference record
pathealth_df = pathealth_df.iloc[:-1]

print('After saving inference record:', pathealth_df.tail(3))



print(pathealth_df.head())
print()
print(len(pathealth_df.columns))
print(pathealth_df.columns)
print()
print('Numerical features:', len(num_features))
print(num_features)
print()
print('Categorical features:', len(cat_features))
print(cat_features)
print()
print('Total records -->', pathealth_df.shape[0])


# <br>The finalized 10 features + 1 target variable for this project:


print(*list(pathealth_df.columns), sep='\n')


# # 2. Preparing data for ML models


# creating split of train, validation, test 60%, 20%, 20% data
from sklearn.model_selection import train_test_split

full_train_df, test_df = train_test_split(pathealth_df, test_size=0.2, random_state=67)

train_df, val_df = train_test_split(full_train_df, test_size=0.25, random_state=67)

# checking the splits
print(len(train_df), len(val_df), len(test_df))
print(round(len(train_df)/len(pathealth_df), 2), round(len(val_df)/len(pathealth_df), 2), round(len(test_df)/len(pathealth_df), 2))



# reseting indices of the split dfs
full_train_df = full_train_df.reset_index(drop=True)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)



# target
y_full_train = full_train_df['insulin'].values
y_train = train_df['insulin'].values
y_val = val_df['insulin'].values
y_test = test_df['insulin'].values



# removing target from feature sets
del full_train_df['insulin']
del train_df['insulin']
del val_df['insulin']
del test_df['insulin']



# importing dict vectorizer
from sklearn.feature_extraction import DictVectorizer



# fn to vectorize dataframes
def vectorize_dfs(train_or_full_df, val_or_test_df):
    
    # initialize dict vectorizer
    dictionary_vectorizer = DictVectorizer(sparse=False)
    
    # create x train or full train after vectorizing features respectively
    X_train_or_full = dictionary_vectorizer.fit_transform(train_or_full_df[cat_features + num_features].to_dict(orient='records'))
    
    # transform the val or test set features respectively
    X_val_or_test = dictionary_vectorizer.transform(val_or_test_df[cat_features + num_features].to_dict(orient='records'))
    
    # extract the featuere names
    feature_names = dictionary_vectorizer.get_feature_names_out()
    
    print('vectorizing successful!')

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


# # 3. Training ML models


# importing the models needed for binary classification problem at hand
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# importing modules related to pipeline, cross validation, metrics 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# creating model pipelines
pipelines = {
    'logistic': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(solver='liblinear', random_state=67))
    ]),
    'random_forest': Pipeline([
        ('model', RandomForestClassifier(random_state=67))
    ]),
    'xgboost': Pipeline([
        ('model', XGBClassifier(eval_metric='logloss', random_state=67))
    ])
}



# hyperparameter grids for cross validation
param_grids = {
    'logistic': {
        'model__C': [0.01, 0.1, 1, 10],
        'model__penalty': ['l1', 'l2'],
    },
    'random_forest': {
        'model__n_estimators': [50, 100, 200],
        'model__max_depth': [None, 10],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2]
    },
    'xgboost': {
        'model__n_estimators': [100, 200],
        'model__max_depth': [4, 6],
        'model__learning_rate': [0.01, 0.1, 0.2],
        'model__gamma': [0, 0.1, 0.2]
    }
}



# stratified KFold with a fixed random state
cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=67)

# fn for training , hyperparameter tuning, cross validation
def train_and_tune(pipelines, param_grids, X_train_or_full, y_train_or_full):
    """Train models with GridSearchCV and return the best estimators."""
    
    # empty dict for best models with best hyperparameters
    best_estimators = {}
    
    # performing hyperparameter tuning using 
    for name in pipelines:
        
        # cross validation and hyperparameter tuning
        print(f"Training and tuning {name}...")
        
        grid = GridSearchCV(estimator=pipelines[name], param_grid=param_grids[name], cv=5, scoring='f1', n_jobs=-1, verbose=1)
        
        grid.fit(X_train_or_full, y_train_or_full)
        
        print(f"Best params for {name}: {grid.best_params_}")
        
        # extracting the best estimators
        best_estimators[name] = grid.best_estimator_
        
        print("-" * 50)
        
    return best_estimators



best_model_params = train_and_tune(pipelines, param_grids, X_train, y_train)


best_model_params


# # 4. Making predictions and evaluations


# # fn to only make predictions and evaluations

# def predict_and_evaluate(best_model_params, X_val_or_test, y_val_or_test):
#     """Use best estimators to predict and print evaluation metrics."""
#     for name, model in best_model_params.items():
        
#         # making the predictions
#         print(f"Evaluating model {name}...")
#         y_pred = model.predict(X_val_or_test)
        
#         # printing the evaluation metrics
#         print(f"Accuracy: {accuracy_score(y_val_or_test, y_pred):.4f}")
#         print(f"Precision: {precision_score(y_val_or_test, y_pred):.4f}")
#         print(f"Recall: {recall_score(y_val_or_test, y_pred):.4f}")
#         print(f"F1 Score: {f1_score(y_val_or_test, y_pred):.4f}")
#         print("-" * 50)




# fn to predict evaluate and plot the evaluation metrics

def predict_and_evaluate(best_model_params, X_val_or_test, y_val_or_test):
    """Use best estimators to predict and print evaluation metrics."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # empty list to store the metrics
    scores = {name: [] for name in best_model_params.keys()}

    # evaluating the metrics
    for name, model in best_model_params.items():
        # making the predictions
        print(f"Evaluating model {name}...")
        y_pred = model.predict(X_val_or_test)

        accuracy = accuracy_score(y_val_or_test, y_pred)
        precision = precision_score(y_val_or_test, y_pred)
        recall = recall_score(y_val_or_test, y_pred)
        f1 = f1_score(y_val_or_test, y_pred)

        # storing the scores
        scores[name] = [accuracy, precision, recall, f1]

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("-" * 50)

    # grouped bar chart  
    model_names = list(scores.keys())
    metric_values = np.array(list(scores.values()))  # shape: (num_models, num_metrics)

    bar_width = 0.2
    indices = np.arange(len(metrics))

    plt.figure(figsize=(10, 6))

    bars = []
    for i, model_name in enumerate(model_names):
        bars.append(plt.bar(indices + i * bar_width, metric_values[i], width=bar_width, label=model_name))

    # value labels for each bar
    for bar_group in bars:
        for bar in bar_group:
            height = bar.get_height()
            plt.annotate(f'{height:.4f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 5),  # 5 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)

    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Comparison of Evaluation Metrics for Models')
    plt.xticks(indices + bar_width * (len(model_names) - 1) / 2, metrics)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.show()



# making predictions on the validation set
predict_and_evaluate(best_model_params, X_val, y_val)



# 
# Since the dataset is slightly imbalanced on comparing with the target variables,
# 
# recall (false negatives) matter much more than precision (false positives) in hyperglycemic patients,
# because people with underlying conditions not treated with increased insulin dosage may result in dire consequences
# 
# Also, precision (false positives) matter much more than recall (false negatives) in hypoglycemic patients,
# since there is an upper limit on what that particular patient's blood glucose level is and what is the current dosage. 
# so significant increase of insulin dosage, on top of previously prescribed insulin dosage in patients carries a huge risk
# 
# Thus, for a start, a combination of F-1 score and as usual accuracy are considered good metrics to select the best model 
# 
# As a result, **xGBoost** produced robust metrics across all 4 evaluation metrics in consideration
# 
# <br>
# 
# Disclaimer: Further analyses and subject matter expertise is needed to come to a definite model selection criteria




# # 5. Best model, full training data train, full hideout data test


print(best_model_params['xgboost'][0])


# instantiating the best model
xgboost_final = best_model_params['xgboost'][0]

# fitting all the training data
xgboost_final.fit(X_full_train, y_full_train)

# make the predictions
y_pred = xgboost_final.predict(X_test)
        
# printing the evaluation metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")



# find out the feature importances 
feature_importances = xgboost_final.feature_importances_

# create a dataframe using feature and importances and sorth it by the most important feature
feature_importance_df = pd.DataFrame({'Feature': feature_names_full_train, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print(feature_importance_df)


plt.figure(figsize=(16, 4))
plt.bar(feature_importance_df['Feature'][:7],feature_importance_df['Importance'][:7])
plt.show()


# only these 7 features are considered by the model. The rest of the features' coefficients are regularized to 0

# # 6. Pickling the best model for inference


# saving the model as a pickled file
import pickle

with open('pickled_model.bin', 'wb') as file:
    pickle.dump(xgboost_final, file)


# initialize final dict vectorizer for inference
dictvec_fulltrain = DictVectorizer(sparse=False)

# create x train or full train after vectorizing features respectively
X_train_full = dictvec_fulltrain.fit_transform(full_train_df.iloc[:, :10].to_dict(orient='records'))
    
# exporting the dict vectorizer
with open('dictvec_fulltrain.bin', 'wb') as file:
    pickle.dump(dictvec_fulltrain, file)


# testing the pickled model
with open('pickled_model.bin', 'rb') as file:
    loaded_model = pickle.load(file)

# You can now use loaded_model to make predictions
y_pred = loaded_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")


# testing the final dict vectorizer
with open('dictvec_fulltrain.bin', 'rb') as file:
    loaded_dictvec = pickle.load(file)

feature_test = loaded_dictvec.transform(test_df.iloc[:, :-1].to_dict(orient='records'))
print(feature_test)

print('best model is selected, pickled and verified!')
