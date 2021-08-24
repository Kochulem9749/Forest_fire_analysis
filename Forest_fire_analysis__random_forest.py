import pandas as pd
import numpy as np
##import matplotlib.pyplot as pp
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
##from keras.utils import to_categorical
import os,sys

Training_Samples = pd.read_csv("E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Outputs_from_R/Training_data.csv")
Prediction_Samples = pd.read_csv("E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Outputs_from_R/Prediction_data.csv")

##Hot_encoding_classes_to have_binary categorical_representation
ord_enc = OrdinalEncoder()
Training_Samples["Burn_Classes"] = ord_enc.fit_transform(Training_Samples[["Classes"]])
##GG=Training_Samples[["Classes", "Burn_Classes"]]


##Create response variable from the hot encoded variable column
Response=Training_Samples["Classes"].values

Predictors = Training_Samples[['Diff_Contrast','Diff_Dissimilarity','Diff_Entropy',
'Diff_Homogeneity','Diff_mean','Diff_Second_moment','Diff_Variance']].values

X_train,X_test,Y_train,Y_test=train_test_split(Predictors,Response,test_size=0.25,random_state=1)

####Automatic search of Hyperparameters
##grid_param = {
##    'n_estimators': [100, 300, 500, 800, 1000,2000],
##    'max_depth':[5,8,15,25,30],
##    'criterion': ['gini', 'entropy'],
##    'min_samples_split':[2,5,10,30],
##    'bootstrap': [True, False]
##}
##
##
##rf_clf = GridSearchCV(estimator=RandomForestClassifier(random_state=1),
##                     param_grid=grid_param,
##                     scoring='accuracy',
##                     cv=3,
##                     n_jobs=-1)
##
##rf_clf.fit(X_train, Y_train)
##best_parameters = rf_clf.best_params_
##print(best_parameters)

###Create a new model by applying the best parameters obtained above
rf_clf = RandomForestClassifier(n_estimators=500, bootstrap='False',criterion='entropy',max_depth=8,min_samples_split=30,n_jobs=-1, random_state=1)

rf_clf.fit(X_train, Y_train)
Y_prediction_RF = rf_clf.predict(X_test)
##
print (metrics.classification_report(Y_test, Y_prediction_RF))
print ("Overall Accuracy:", round(metrics.accuracy_score(Y_test, Y_prediction_RF),3))

#Calculating the accuracy measures\n",
##mape = np.mean(np.abs((Y_test - Y_prediction_RF)/Y_test))*100
##r_squared= r2_score(Y_test, Y_prediction_RF)
###printing the accuracy measures, rounded to 4 dp,
##print(mape)

mae = metrics.mean_absolute_error(Y_test, Y_prediction_RF)
mse = metrics.mean_squared_error(Y_test, Y_prediction_RF)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_prediction_RF)
print(r2)

Covariates= Prediction_Samples[['Diff_Contrast','Diff_Dissimilarity','Diff_Entropy',
'Diff_Homogeneity','Diff_mean','Diff_Second_moment','Diff_Variance']].values

Fire_prediction_RF = rf_clf.predict(Covariates)

grid_coordinates=Prediction_Samples[['X','Y']]



##fn=rf_predictors.reshape(val1.shape[1],val1.shape[2])
predi_df=pd.DataFrame(Fire_prediction_RF)
final_rf=pd.concat([grid_coordinates,predi_df],axis=1)
final_rf.columns=['x_coord','y_coord','Fire_prediction_RF']

final_rf.to_csv("E:/Masters_Program/Semester 2/DIP/Images/Forest_fire_Analysis/Outputs_from_R/RFPredicted_Fire_spread.csv")

