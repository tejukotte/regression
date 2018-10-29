import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from pandas import plotting as plt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot

import numpy as np

from sklearn import preprocessing as pp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics

# Clean the violent crimes dataset
def crimesClean():

    # read the original CSV file with uncleaned data and put it into a pandas dataframe
    df = pd.read_csv("violentcrimes.csv")

    # put the names of the columns into a list to be used with melt
    columnList = df.columns.tolist()

    # Use melt to change the table from wide to long format.
    df_melted = pd.melt(df, id_vars=['Region', 'Division', 'State'], value_vars=columnList[3:(len(columnList)+1)],
                        var_name='Year', value_name='Crimes')
    # outputs the long format table to a new csv
    df_melted.to_csv('vcOut.csv', index=False)


def imputeVC():
    
    #read the file we finished with from before
    df = pd.read_csv('vcOut.csv', na_values='null', header=0, index_col=None)
    
    plt.scatter_matrix(df)
    
    #plot the original for reference 
    plot.figure('original')
    plot.hist(df['Year'])    
    
    
    #Feature Scaling
    scale = pp.MinMaxScaler()
    
    #using the columns that have numberical values
    columns = ['Year', 'Crimes']
    
    #use the scale variable from above to change the columns to a better scale
    df[columns] = scale.fit_transform(df[columns])
    
    print('\nScaled:\n', df.head())
    
    #plot the scaled example
    plot.figure('scaled')
    plot.hist(df['Year'])
    plot.show()    
    plt.scatter_matrix(df)
    
    #create dummy variables; used for the region, division and state
    df = pd.get_dummies(df, columns = ['Region', 'Division', 'State'], prefix = '', prefix_sep = '')
    
    #method for avoiding colliniarity trap, deleted one from each column that dummies were created for
    #df = df.drop(["South", "East South Central"], axis=1)
    
    print('\nDummy Variable Encoding:\n', df.head())    
    
       
    #plt.scatter_matrix(df) 
    
    np.random.seed(100)
    no_null = df.dropna(how='any')
    y = no_null['Crimes']
    x = no_null.drop(['Crimes', 'Wyoming'], axis=1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # build regression model on training data ###################
    lm_test = LinearRegression().fit(x_train, y_train)
    
    # evaluate the model on the testing data to get R2 ##########
    ''' R2 = percent of variance of y explained by model '''
    r2 = lm_test.score(x_test, y_test)
    print('\nHold Out R2:', round(r2, 4))
    
    # get predictions and compare with actual to get RMSE #######
    # Make predictions using the testing set
    pred = lm_test.predict(x_test)
    
    # Get RMSE from predictions & actual values (y) from testing set
    ''' RMSE: average difference of predicted vs. actual in Y units'''
    # RMSE is the square root of MSE
    rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
    print('Hold Out RMSE:', round(rmse, 4))
    
      
    # create an empty linear regression object
    lm_k = LinearRegression()
    
    # get the R2 and MSE for each model generated with k-fold (number of models = k) ##################
    # the default for cross_val score is R2, take 10 folds b/c common in literature
    scores_r2 = list(np.round(cross_val_score(lm_k, x, y, cv=10), 4))
    # set scoring parameter to get neg mse (use 10 folds for everything to keep consistency)
    scores_mse = cross_val_score(lm_k, x, y, cv=10, scoring="neg_mean_squared_error")
    # to get rmse we need to take the square root of the absolute mse values
    scores_rmse = list(np.round(np.sqrt(np.abs(scores_mse)), 4))
    print('\nCross-validated R2 By Fold:\n', scores_r2, "\nCross-Validated MSE By Fold:\n", scores_rmse)
    
    # Get the overall R2 and MSE for this type of model on the data ####################################
    # generate a prediction list
    predictions = cross_val_predict(lm_k, x, y, cv=10)
    
    # compare prediction vs actual values to get metrics
    # r2
    r2 = metrics.r2_score(y, predictions)
    # take square root of mse to get rmse
    rmse = np.sqrt(metrics.mean_squared_error(y, predictions))
    print('Cross-Validated R2 For All Folds:', round(r2, 4), '\nCross-Validation RMSE For All Folds:', round(rmse,4) )
    
    # Build Final Model  #############################################################################################
    # We want to build the final model with all the data
    lm_final = LinearRegression().fit(x, y)
    
    
    # Prep Data for Output ###########################################################################################
    # Predict With Model to Fill Null Values ######################################
    
    
    # get the row number (index #) of null values in crime rate
    null_list = df.index[df['Crimes'].isnull()].tolist()
    
    # get the x values for use in the prediction of null divorce rates (everything but divorce rate & dropped dummy)
    pred_vals = df.drop(['Crimes', 'Wyoming'], axis=1)
    
    # for each row where divorce rate is null, set divorce rate to be the prediction generated by model
    # df and pred_values will have the same index, so can access the corresponding data from row position
    for x in null_list:                 # x is the position of the null value row
        # go to row x and col div_rate in df, replace value with predictions generated from corresponding x vals
        df.ix[x, 'Crimes'] = lm_final.predict([pred_vals.iloc[x].tolist()])
    
    
    # Reshape to Remove Dummies ###################################################
    # get only the dummy columns and set them to a new dataframe (row location is preserved)
    #state = df.drop(df.columns[:3], axis=1)
    
    # remove all dummy columns from df dataframe
    #df.drop(df.columns[3:], axis=1, inplace=True)
    
    # create new variable state that is the column name of the highest value (1) for each row of dummies
    df['State'] = state.idxmax(axis=1)   # axis = 1 tells it to get the column name
    print('\nFinal Dataset:\n',df.head())
    
    
    
    
    df.to_excel('Regression5.2.xls', index=False, na_rep='null')
    
    
    
    