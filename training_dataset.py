import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from numpy import mean
from numpy import std
from scipy import stats
import warnings
# from pandas.core.common import SettingWithCopyWarning

def remove_outliers_GUI_Screens(X):
	flag_value = 0
	zscore = stats.zscore(X['GUI_screens'])
	for i in range(0, X.shape[0]):
		if(abs(zscore[i])>3):
			X.iloc[i]['GUI_screens']= 0
			flag_value = 1
	if flag_value==1:
		maximum_value = X['GUI_screens'].max()
		X['GUI_screens']=X['GUI_screens'].replace(0,maximum_value)
	return(X)

def remove_outliers_manual_steps(X):
	flag_value = 0
	zscore = stats.zscore(X['manual_steps'])
	for i in range(0, X.shape[0]):
		if(abs(zscore[i])>3):
			X.iloc[i]['manual_steps']= 0
			flag_value = 1
	if flag_value==1:
		maximum_value = X['manual_steps'].max()
		X['manual_steps']=X['manual_steps'].replace(0,maximum_value)
	return(X)

def remove_outliers_transaction_time(X):
	flag_value = 0
	zscore = stats.zscore(X['transaction_time'])
	for i in range(0, X.shape[0]):
		if(abs(zscore[i])>3):
			X.iloc[i]['transaction_time']= 0
			flag_value = 1
	if flag_value==1:
		maximum_value = X['transaction_time'].max()
		X['transaction_time']=X['transaction_time'].replace(0,maximum_value)
	return(X)

def remove_outliers_volume(X):
	flag_value = 0
	zscore = stats.zscore(X['volume'])
	for i in range(0, X.shape[0]):
		if(abs(zscore[i])>3):
			X.iloc[i]['volume']= 0
			flag_value = 1
	if flag_value==1:
		maximum_value = X['volume'].max()
		X['volume']=X['volume'].replace(0,maximum_value)
	return(X)

def remove_outliers_FTE(X):
	flag_value = 0 
	zscore = stats.zscore(X['FTE'])
	for i in range(0, X.shape[0]):
		if(abs(zscore[i])>3):
			X.iloc[i]['FTE']= 0
			flag_value = 1
	if flag_value==1:
		maximum_value = X['FTE'].max()
		X['FTE']=X['FTE'].replace(0,maximum_value)
	return(X)

def remove_outliers_SLA(X):
	flag_value = 0
	zscore = stats.zscore(X['SLA'])
	for i in range(0, X.shape[0]):
		if(abs(zscore[i])>3):
			X.iloc[i]['SLA']= 0
			flag_value = 1
	if flag_value==1:
		maximum_value = X['SLA'].max()
		X['SLA']=X['SLA'].replace(0,maximum_value)
	return(X)

def remove_outliers_Decision_Points(X):
	flag_value = 0
	zscore = stats.zscore(X['Decision_points'])
	for i in range(0, X.shape[0]):
		if(abs(zscore[i])>3):
			X.iloc[i]['Decision_points']= 0
			flag_value=1
	if flag_value==1:
		maximum_value = X['Decision_points'].max()
		X['Decision_points']=X['Decision_points'].replace(0,maximum_value)
	return(X)

def remove_outliers_Business_exception(X):
	flag_value=0
	zscore = stats.zscore(X['Business_exception'])
	for i in range(0, X.shape[0]):
		if(abs(zscore[i])>3):
			X.iloc[i]['Business_exception']= 0
			flag_value=1
	if flag_value==1:
		maximum_value = X['Business_exception'].max()
		X['Business_exception']=X['Business_exception'].replace(0,maximum_value)
	return(X)

def main():
	# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
	use_case = pd.read_csv("dataset_complexity.csv")
	X = use_case[['GUI_screens','manual_steps','transaction_time','volume','FTE','SLA','Decision_points','Business_exception']]
	X = remove_outliers_GUI_Screens(X)
	X = remove_outliers_manual_steps(X)
	X = remove_outliers_transaction_time(X)
	X = remove_outliers_volume(X)
	X = remove_outliers_FTE(X)
	X = remove_outliers_SLA(X)
	X = remove_outliers_Decision_Points(X)
	X = remove_outliers_Business_exception(X)
	X.describe().to_csv("Detailed Description.csv")
	X.to_csv('Intermediate_data.csv')
	#min_max_scaler = MinMaxScaler()
	#X[['GUI_screens','manual_steps','transaction_time','volume','FTE','SLA','Decision_points','Business_exception']] = min_max_scaler.fit_transform(X[['GUI_screens','manual_steps','transaction_time','volume','FTE','SLA','Decision_points','Business_exception']])
	#X.to_csv('transformed_data.csv')
	print("Data Has Been Cleaned. Run Successful")
	'''print('mean=%.3f stdv=%.3f' % (mean(X.volume), std(X.volume)))
	data_mean = mean(X.volume)
	data_std = std(X.volume)
	cut_off = data_std * 3
	lower, upper = data_mean - cut_off, data_mean + cut_off
	outliers = [x for x in X.volume if x < lower or x > upper]
	print('Identified outliers: %d' % len(outliers))'''
	#print(stats.zscore(X['volume']))



if __name__ == "__main__":
    main()