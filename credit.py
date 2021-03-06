# %%
%%capture
# Load packages
# Import 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer
import time

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import plot_partial_dependence
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.utils import resample,shuffle
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE

# Others
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')

cmap=sns.color_palette('Blues_r')

# %%
# Load data from the csv file
df = pd.read_csv('loan_data.csv', index_col=None)

# Change the dots in the column names to underscores
df.columns = [c.replace(".", "_") for c in df.columns]
print(f"Number of rows/records: {df.shape[0]}")
print(f"Number of columns/variables: {df.shape[1]}")
df.head()

#%%
# Create train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=0)
train, valid = train_test_split(df, test_size=0.2, random_state=0)
train.head()

#%%
# Train: X and y split
X_train = train.drop('not_fully_paid', axis=1)
y_train = train[['not_fully_paid']]
X_train.head()

#%%

# Train: X and y split
X_test = test.drop('not_fully_paid', axis=1)
y_test = test[['not_fully_paid']]
X_test.head()

#%%

# Train: X and y split
X_valid = valid.drop('not_fully_paid', axis=1)
y_valid = test[['not_fully_paid']]
X_valid.head()

# Check for missing values
df.info()

# Split col by data type
num = ['credit_policy', 'int_rate', 'installment', 'log_annual_inc', 'dti', 'fico', 'days_with_cr_line', 'revol_bal', 'revol_util', 'inq_last_6mths', 'delinq_2yrs', 'pub_rec']
non_num = ['purpose']

#%%

# Check distribution of num values
X_train[num].describe()

# Check distribution (num)
print(f'=== Distribution of features (before log transformation) ===')
plt.figure(figsize=(15, 18))
for i, col in enumerate(num):
    # Plot distribution 
    plt.subplot(4,3,i+1); sns.distplot(X_train[col], color='blue')
    plt.title(f'Distribution of {col}')
# Show the plot
plt.tight_layout()
plt.show()

# Log transform function
def log_transform(data, to_log):
    X = data.copy()
    for item in to_log:
        # Add 1 to the data to prevent infinity values
        X[item] = np.log(1+X[item])
    return X

# Display unique values in cat data
X_train['purpose'].unique()

# Transform categorial features to num 
X_train_e = pd.get_dummies(data=X_train)
X_train_e = X_train_e.drop(['purpose_all_other'], axis=1)

X_test_e = pd.get_dummies(data=X_test)
X_test_e = X_test_e.drop(['purpose_all_other'], axis=1)

X_valid_e = pd.get_dummies(data=X_valid)
X_valid_e = X_valid_e.drop(['purpose_all_other'], axis=1)

# Log transform
to_log = ['credit_policy', 'int_rate', 'installment', 'dti', 'fico', 'days_with_cr_line', 'revol_bal', 'revol_util', 'inq_last_6mths', 'delinq_2yrs', 'pub_rec']
X_train_e_l = log_transform(X_train_e, to_log)
X_test_e_l = log_transform(X_test_e, to_log)
X_valid_e_l = log_transform(X_valid_e, to_log)
X_train_e_l.head()

# Check distribution (num)
print(f'=== Distribution of features (after log transformation) ===')
plt.figure(figsize=(15, 18))
for i, col in enumerate(num):
    # Plot distribution 
    plt.subplot(4,3,i+1); sns.distplot(X_train_e_l[col], color='blue')
    plt.title(f'Distribution of {col}')
# Show the plot
plt.tight_layout()
plt.show()

#%%

# Initialize a standard scaler and fit it
scaler = StandardScaler()
scaler.fit(X_train_e_l)

# Scale and center the data
tmp_train = scaler.transform(X_train_e_l)
tmp_test = scaler.transform(X_test_e_l)
tmp_valid = scaler.transform(X_valid_e_l)

# Create a pandas DataFrame
X_train_e_l_n = pd.DataFrame(data=tmp_train, index=X_train_e.index, columns=X_train_e.columns)    
X_test_e_l_n = pd.DataFrame(data=tmp_test, index=X_test_e.index, columns=X_test_e.columns) 
X_valid_e_l_n = pd.DataFrame(data=tmp_valid, index=X_valid_e.index, columns=X_valid_e.columns) 

X_train_e_l_n.describe()

X_train_e_l_n.head()

#%%

# Check distribution (non-num)
plt.figure(figsize=(8, 8))
# Plot distribution 
for i, col in enumerate(['purpose']):
    order = train[col].value_counts().index   
    plt.subplot(1,1,i+1); sns.countplot(train[col],palette=cmap, order = order ) 
    plt.title(f'Distribution of {col}')
    plt.ylabel('No. of loans')
    plt.xticks(rotation=90)
# Show the plot
plt.tight_layout()
plt.show()

X_train[['purpose']].describe()

# Plot tsne scatter plot
def tsne_scatterplot(data, hue):
    # Color the points 
    sns.scatterplot(x="t_SNE_PC_1", y="t_SNE_PC_2", hue=hue, data=data, alpha=0.3)
    plt.title(hue)

# Create a t-SNE model with learning rate 50
m = TSNE(learning_rate=50)

X_train_num_l_n = X_train_e_l_n[num]
X_train_num_l_n.head()

# Fit and transform the 
tsne_features = m.fit_transform(X_train_num_l_n)
print(tsne_features.shape)

#%%

# Create df
df_tsne = y_train.copy()
df_tsne = pd.concat([X_train, y_train], axis=1)
df_tsne['t_SNE_PC_1'] = tsne_features[:,0]
df_tsne['t_SNE_PC_2'] = tsne_features[:,1]
df_tsne.head()

#%%

# Viz of y
plt.figure(figsize=(8,8))

# y 
tsne_scatterplot(df_tsne, 'not_fully_paid')
# Plot
plt.show()

#%%

# Viz of purpose
plt.figure(figsize=(8,8))

# y 
tsne_scatterplot(df_tsne, 'purpose')
# Plot
plt.show()

#%%

# Check distribution (non-num)
plt.figure(figsize=(8, 8))
# Plot distribution 
for i, col in enumerate(['not_fully_paid']):
    order = train[col].value_counts().index   
    plt.subplot(1,1,i+1); sns.countplot(train[col],palette=cmap, order = order ) 
    plt.title(f'Distribution of {col}')
    plt.ylabel('No. of clients')
    plt.xticks(rotation=90)
# Show the plot
plt.tight_layout()
plt.show()

#%%

def stack_bar(d, xlabel, hue = 'lead_type'):
    plt.figure(figsize=(5, 5))
    train_pct = (d.groupby([xlabel,hue])['credit_policy'].count()/d.groupby([xlabel])['credit_policy'].count())
    train_pct.unstack().plot.bar(stacked=True)
    plt.ylabel('%') 
    plt.title(f'100% stack bar chart \n By {xlabel}')
    plt.show()   

stack_bar(train, 'purpose', 'not_fully_paid')

#%%

# Convert target var to a categorial var
train['y'] = np.where(train['not_fully_paid']==1, 'yes', 'no')
train.info()

# Not fully paid 
train[train['y']=='yes'].describe()

#%%

# Check distribution (num)
plt.figure(figsize=(10, 30))
# Plot distribution 
for i, col in enumerate(num):
    plt.subplot(13,1,1+i); sns.violinplot(x=col, y='y', data=train, inner=None, color='lightgray');sns.stripplot(x=col, y='y', data=train, size=0.8,jitter=True);
    plt.title('Distribution of target variable');
    plt.ylabel('Not fully paid')
    # Show the plot
plt.tight_layout()
plt.show()

#%%

def upsample_data(X, y):
    # Concat
    data = pd.concat([X, y], axis=1)
    #data.head()
    subscribe = data[data['not_fully_paid']==1]
    not_subscribe = data[data['not_fully_paid']==0]
    # Upsample minority and combine with majority
    data_upsampled = resample(not_subscribe, replace=True, n_samples=len(subscribe), random_state=0)
    upsampled = pd.concat([subscribe, data_upsampled])
    # Upsampled feature matrix and target array
    X_up = upsampled.drop('not_fully_paid', axis=1)
    y_up = upsampled[['not_fully_paid']]
    #X_up.head()
    
    return X_up, y_up

# Perform upsample on all dataset
X_train_e_l_n_up, y_train_up = upsample_data(X_train_e_l_n, y_train)
X_test_e_l_n_up, y_test_up = upsample_data(X_test_e_l_n, y_test)
X_valid_e_l_n_up, y_valid_up = upsample_data(X_valid_e_l_n, y_valid)

X_train_e_l_n_up.info()

#%%

# Set attributes for PCA analysis
n=18
columns=['PCA_1', 'PCA_2', 'PCA_3', 'PCA_4', 'PCA_5', 'PCA_6', 'PCA_7', 'PCA_8', 'PCA_9', 'PCA_10', 'PCA_11', 'PCA_12', 'PCA_13', 'PCA_14', 'PCA_15', 'PCA_16', 'PCA_17', 'PCA_18']

# Create the PCA instance and fit and transform the data with pca
pca = PCA(n_components=n)
pc = pca.fit_transform(X_train_e_l_n)
df_pc = pd.DataFrame(pc, columns=columns, index=X_train_e_l_n.index)
df_pc.head()

pca.explained_variance_ratio_

# PCA df which store PCA componenets and corresponding y
df_PCA = pd.concat([df_pc, y_train], axis=1)
df_PCA.head()

#%%

targets = [0, 1]
colors = ['orange', 'blue']

fig, ax = plt.subplots(figsize=(8,8))


# For loop to create plot
for target, color in zip(targets,colors):
    indicesToKeep = df_PCA['not_fully_paid'] == target
    ax.scatter(df_PCA.loc[indicesToKeep, 'PCA_1']
               , df_PCA.loc[indicesToKeep, 'PCA_2']
               , c = color
               , s = 50
              , alpha =0.2)

# Legend    
ax.legend(targets)
ax.grid()
plt.show()

#%%

# Instantiate
pca = PCA(n_components=n)

# Fit and transform
principalComponents = pca.fit_transform(X_train_e_l_n)

# List principal components names
principal_components =columns

# Create a DataFrame
pca_df = pd.DataFrame({'Variance Explained': pca.explained_variance_ratio_,
             'PC':principal_components})

plt.figure(figsize=(8, 8))
plt.title('PCA - explained variance ratio')
# Plot DataFrame
sns.barplot(x='PC',y='Variance Explained', 
           data=pca_df, color="c")
plt.xticks(rotation=45)
plt.show()

#%%

lt.figure(figsize=(8, 8))
# Instantiate, fit and transform
#pca2 = PCA()
#principalComponents2 = pca2.fit_transform(X_train_valid_e_n_up)

# Assign variance explained
var = pca.explained_variance_ratio_

# Plot cumulative variance
cumulative_var = np.cumsum(var)*100
plt.plot(cumulative_var,'k-o',markerfacecolor='None',markeredgecolor='k')
plt.title('Principal Component Analysis \n Cumulative Proportion of Variance Explained',fontsize=12)
plt.xlabel("Principal Component",fontsize=12)
plt.ylabel("Cumulative Proportion of Variance ",fontsize=12)
plt.show()

#%%

# Function to find lowest C and plot the result
def lowest_err(model, C_values, train_errs, valid_errs):
    
    # Print lowest valid err 
    C_values_df = pd.DataFrame(C_values)
    valid_errs_df = pd.DataFrame(valid_errs)
    min_err = min(valid_errs_df[0])
    min_idx = valid_errs_df[0].idxmin(axis = 1)
    min_C = C_values_df.loc[min_idx, 0]
    print(f'Min validation error ({model})  occur at C={round(min_C, 5)} and error = {round(min_err,5)}')

    # Plot results
    plt.figure(figsize=(8,5))
    plt.semilogx(C_values, train_errs, C_values, valid_errs)
    plt.annotate(f'Min error occur at Hyper-param ={round(min_C, 5)} and error = {round(min_err,5)}', xy=(min_C, min_err),fontsize=12,arrowprops=dict(facecolor='red', shrink=0.5),)
    plt.legend(("train", "validation"))
    plt.title(model)
    plt.xlabel('Hyper-parameter')
    plt.ylabel('Error')
    plt.show()
    
    # Return
    return {'Model':model, 'Best hyperparam': min_C}

#%%
# A logistic regression 
min_C = list()

# Train and validaton errors initialized as empty list
train_errs = list()
valid_errs = list()
C_values = np.logspace(-6, -2, 21)

# Loop over values of C_value
for C_value in C_values:
    # Create LogisticRegression object and fit
    clf = LogisticRegression(C=C_value)
    clf.fit(X_train_e_l_n_up, y_train_up)
    
    # Evaluate error rates and append to lists
    train_errs.append( 1.0 - clf.score(X_train_e_l_n_up, y_train_up) )
    valid_errs.append( 1.0 - clf.score(X_valid_e_l_n_up, y_valid_up) )

# Plot and find min C
min_C.append( lowest_err('Logistic regression', C_values, train_errs, valid_errs) )

#%%
# B gradient boosting
# Train and validaton errors initialized as empty list
train_errs = list()
valid_errs = list()
C_values = [int(x) for x in np.linspace(1, 900, 21)]

# Loop over values of C_value
for C_value in C_values:
    # Create LogisticRegression object and fit
    clf = GradientBoostingClassifier(n_estimators=C_value)
    clf.fit(X_train_e_l_n_up, y_train_up)
    
    # Evaluate error rates and append to lists
    train_errs.append( 1.0 - clf.score(X_train_e_l_n_up, y_train_up) )
    valid_errs.append( 1.0 - clf.score(X_valid_e_l_n_up, y_valid_up) )

# Plot and find min C
min_C.append( lowest_err('Gradient boosting', C_values, train_errs, valid_errs) )

#%%

# C random forest
# Train and validaton errors initialized as empty list
train_errs = list()
valid_errs = list()
C_values = [int(x) for x in np.linspace(1, 900, 21)]

# Loop over values of C_value
for C_value in C_values:
    # Create LogisticRegression object and fit
    clf = RandomForestClassifier(n_estimators=C_value,random_state=0)
    clf.fit(X_train_e_l_n_up, y_train_up)
    
    # Evaluate error rates and append to lists
    train_errs.append( 1.0 - clf.score(X_train_e_l_n_up, y_train_up) )
    valid_errs.append( 1.0 - clf.score(X_valid_e_l_n_up, y_valid_up) )

# Plot and find min C
min_C.append( lowest_err('Random Forest', C_values, train_errs, valid_errs) )

#%%

# Print min C
min_C = pd.DataFrame(min_C)
min_C.set_index('Model', inplace=True)
min_C

# Refit model with both train and valid data
X_train_valid_e_l_n_up = pd.concat([X_train_e_l_n_up, X_valid_e_l_n_up], axis=0)
X_train_valid_e_l_n_up.tail()

y_train_valid_e_l_n_up = pd.concat([y_train_up,y_valid_up], axis=0)
y_train_valid_e_l_n_up.tail()

# Set RFE params
n_features_to_select = 12
step = 1

# RFE with Logistic regression
rfe_lg = RFE(estimator=LogisticRegression(C=min_C.loc['Logistic regression', 'Best hyperparam']), 
             n_features_to_select=n_features_to_select, step=step, verbose=1)
rfe_lg.fit(X_train_valid_e_l_n_up, y_train_valid_e_l_n_up)
# Calculate the score on the test set
#score = rfe_lg.score(X_test_e_l_n, y_test)
score = rfe_lg.score(X_test_e_l_n_up, y_test_up)
print('Logistic regression can explain {0:.1%} of the variance in the test set'.format(score))
# Assign the support array to mask
lg_mask = rfe_lg.support_

# RFE with Gradient boosting 

# Select 10 features with RFE on a GradientBoostingClassifier, drop 1 features on each step
rfe_gb = RFE(estimator=GradientBoostingClassifier(n_estimators=int(min_C.loc['Gradient boosting', 'Best hyperparam']), random_state=0), 
             n_features_to_select=n_features_to_select, step=step, verbose=1)
rfe_gb.fit(X_train_valid_e_l_n_up, y_train_valid_e_l_n_up)
# Calculate the score on the test set
#score = rfe_gb.score(X_test_e_l_n, y_test)
score = rfe_gb.score(X_test_e_l_n_up, y_test_up)
print('Gradient boosting can explain {0:.1%} of the variance in the test set'.format(score))
# Assign the support array to mask
gb_mask = rfe_gb.support_

# RFE with RandomForestClassifier
rfe_rf = RFE(estimator=RandomForestClassifier(n_estimators=int(min_C.loc['Random Forest', 'Best hyperparam']), random_state=0), 
             n_features_to_select=n_features_to_select, step=step, verbose=1)
rfe_rf.fit(X_train_valid_e_l_n_up, y_train_valid_e_l_n_up)
# Calculate the score on the test set
#score = rfe_rf.score(X_test_e_l_n_up, y_test_up)
#score = rfe_rf.score(X_test_e_l_n, y_test)
score = rfe_rf.score(X_test_e_l_n_up, y_test_up)
print('Random Forest can explain {0:.1%} of the variance in the test set'.format(score))
# Assign the support array to mask
rf_mask = rfe_rf.support_

# Sum the votes of the three models
votes = np.sum([lg_mask, rf_mask, gb_mask], axis=0)
# Create a mask for features selected by all 3 models
meta_mask = votes >= 2
print('The following features are selected by >= 2 models:')
X_train_valid_e_l_n_up.loc[:, meta_mask].columns

# Apply dimensionality reduction on X
X_train_mask = X_train_valid_e_l_n_up.loc[:, meta_mask]
X_test_mask = X_test_e_l_n.loc[:, meta_mask]
X_train_mask.head()

#%%

# Perform random forest
rf = RandomForestClassifier(n_estimators=int(min_C.loc['Random Forest', 'Best hyperparam']), random_state=0)
rf.fit(X_train_mask, y_train_valid_e_l_n_up)
score = rf.score(X_test_mask, y_test)
print('Random forest (dimension reduced) can explain {0:.1%} of the variance in the test set.'.format(score))
# Predict the labels of the test data: y_pred
y_pred = rf.predict(X_test_mask)
# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Perform gradient boosting
gb = GradientBoostingClassifier(n_estimators=int(min_C.loc['Gradient boosting', 'Best hyperparam']), random_state=0)
gb.fit(X_train_mask, y_train_valid_e_l_n_up)
score = gb.score(X_test_mask, y_test)
print('Gradient boosting (dimension reduced) can explain {0:.1%} of the variance in the test set.'.format(score))
# Predict the labels of the test data: y_pred
y_pred = gb.predict(X_test_mask)
# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%

# Retrieve feature importance from the tree-based models
col_names = X_test_mask.columns
model_rank = list()
def model_impt(model, label):
    tmp = dict()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    #print(importances)
    # Print the feature ranking
    print(f"Feature ranking ({label}):")
    tmp['Model'] = label
    for f in range(len(indices)):
        print("%d. %s (%f)" % (f + 1, col_names[indices[f]], importances[indices[f]]))
        tmp[col_names[indices[f]]] = importances[indices[f]]
    print (' ')
    return tmp

# Random forest
model_rank.append( model_impt(rf, 'Random forest'))
# Gradient boosting
model_rank.append( model_impt(gb, 'Gradient boosting'))

# Print min C
model_rank = pd.DataFrame(model_rank)
model_rank.set_index('Model', inplace=True)
model_rank

#%%

# Correlation of numerical attributes
plt.figure(figsize=(10, 2))
plt.title('Feature importance')
sns.heatmap(model_rank, annot=True, fmt='.2f', cmap='RdYlGn')
plt.yticks(rotation=360)
plt.show()

#%%

# Partial plot
fig, ax = plt.subplots(figsize=(20, 30))
ax.set_title("Random forest Model - Partial plot")
#tree_disp = plot_partial_dependence(rfe_rf, X_train, list(range(0, 18)), ax=ax)
tree_disp = plot_partial_dependence(rf, X_train_mask, list(range(0, 11)), ax=ax)

#%%

# Correlation of numerical attributes
plt.figure(figsize=(8, 8))
plt.title('Correlation of important features')
sns.heatmap(X_train_mask.corr(), annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()

#%%

# Define list to store result 
final_result = list()
# Define list to store result 
final_model = list()
def model_run(model, clf, X=X_test, y=y_test):

    # Evaluate error rates and append to lists
    y_test_predict = clf.predict(X)
    auc = roc_auc_score(y, y_test_predict)
    precision =precision_score(y, y_test_predict)
    recall = recall_score(y, y_test_predict)
    accuracy = clf.score(X, y)

    # Print classification report
    print(f'== {model} - Classification report ==')
    print(classification_report(y, y_test_predict))
    
    # Plot confusion matrix
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(confusion_matrix(y, y_test_predict)), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title(f'{model}- Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')   
    plt.show()
    
    # Plot ROC curve
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X)[:,1]
    # Generate ROC curve values: fpr, tpr, thresholds
    fpr, tpr, thresholds= roc_curve(y_test, y_pred_prob)
    # Plot ROC curve
    plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    plt.plot(fpr, tpr, marker='.', label=model)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model} - ROC curve')
    plt.annotate(f'AUC ={round(auc, 5)} ', xy=(0.7, 0.5),fontsize=12,)
    # show the legend
    plt.legend()    
    plt.show()
    
    # Plot precision recall curve
    precision_, recall_, _ = precision_recall_curve(y_test, y_pred_prob)
    # Random classifier is one which randomly predicts a class and is the ratio of the ones in the dataset
    random_clf = y_test[y_test==1].count() / len(y_test) #len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [random_clf, random_clf], linestyle='--', label='Average precision')
    plt.plot(recall_, precision_, marker='.', label=model)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model} - Precision recall curve')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    return {'Model':model, \
            'Accuracy': round(accuracy,5) , \
            'Precision':round(precision,5) , \
            'Recall':round(recall,5), \
            'AUC':round(auc,5)}

#%%

# Logistic regression v1 (baseline)
clf = LogisticRegression()
clf.fit(X_train_e_l_n, y_train)
final_model.append(model_run('Logistic regression (feature selected)', clf, X=X_test_e_l_n, y=y_test))

#%%

# Initialise kfold
kf = KFold(n_splits=5, random_state=42, shuffle=True)
def best_param_search(model, pipe, new_params,kf):
    # Run search for best param

    clf = GridSearchCV(pipe, param_grid=new_params, cv=kf, scoring='roc_auc',
                            return_train_score=True)
    # Fit to model with best hyper-param
    clf.fit(X_train_e_l_n, y_train)
    # Print best results
    clf.cv_results_['mean_test_score'], clf.cv_results_['mean_train_score']
    # Best score 
    clf.best_score_
    # Best param
    clf.best_params_
    # Print test result
    y_test_predict = clf.best_estimator_.predict(X_test_e_l_n)
    score = roc_auc_score(y_test, y_test_predict)
    
    # Print all stats
    # final_model.append(model_run(model, clf, X=X_test, y=y_test))

    #return clf.best_params_, score 
    return clf, {'Model':model, 'Best hyperparam': clf.best_params_, 'AUC':round(score,5)}
    #return {'Model':model, 'AUC':round(score,5)}

#%%

# Logistic regression 
penalty = ['l1','l2','elasticnet','none']
c_values = np.logspace(-7, 1, 10)
# Define params
params = {
    'penalty': penalty,
    'C': c_values,
    'random_state': [0]
}
new_params = {'logisticregression__' + key: params[key] for key in params}

# Create pipeline
pipe = make_pipeline(SMOTE(random_state=42), 
                              LogisticRegression(penalty='l1' ,C=0, random_state=0))
# Search for best param
lr_clf, lr = best_param_search('Logistic Regression (improved)', pipe, new_params,kf)
# Print result
final_model.append(model_run('Logistic Regression (improved)', lr_clf, X=X_test_e_l_n, y=y_test))
# Append result
final_result.append(lr)
final_result
final_model

#%%

### Random forest 
n_estimators = np.linspace(600,800, num=10)
# Define rf params
params = {
    'n_estimators': [int(x) for x in n_estimators],
    'max_depth': [5, 15, 20],
    'random_state': [0]
}
new_params = {'randomforestclassifier__' + key: params[key] for key in params}
# Create pipeline
pipe = make_pipeline(SMOTE(random_state=42), 
                              RandomForestClassifier(n_estimators=375, random_state=0))
# Search for best param
rf_clf, rf = best_param_search('Random forest (improved)', pipe, new_params,kf)
# Print result
final_model.append(model_run('Random forest (improved)', rf_clf, X=X_test_e_l_n, y=y_test))

# Append result
final_result.append(rf)
final_result
final_model

#%%

# Present result in table 
final_model_df = pd.DataFrame(final_model)
final_model_df.set_index('Model', inplace=True)
#final_model_df.head()

#%%

# Print model label
index = ['Model 1','Model 2','Model 3']
conclusion = final_model_df.copy()
conclusion['No.'] = index
conclusion.reset_index(inplace=True)
conclusion.set_index('No.',inplace=True)
conclusion

# Rename col
conclusion.rename(columns={'Model':'Description'}, inplace=True)
#conclusion
conclusion
