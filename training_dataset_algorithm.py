'''
    File name: training_dataset_algorithm.py
    Author: Indrani Chakraborty
    Date created: 1/09/2021
    Date last modified: 17/09/2021
    Python Version: 3.6.8
    Description: This is the header for training_dataset_algorithm.py. It aims at Clustering Data Points entered by User for Complexity Assessment

'''

import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from numpy import mean
from numpy import std
from scipy import stats
import warnings
# from pandas.core.common import SettingWithCopyWarning
from sklearn.decomposition import PCA
from training_dataset import *
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn import svm

#Creating a Function which takes the number of GUI Screens and the Processed Column of GUI Screens and the maximum value of the scaled data and is able to scale the input from User and Return the Scaled Data

def clean_live_data_GUI_Screens(b,column,max_b):
    max_value = column.max()
    min_value = column.min()
    gui_input_scaled = (b-min_value)/(max_value-min_value)
    if gui_input_scaled>0:
        if gui_input_scaled>max_b:
            gui_input_scaled = max_b
        else:
            gui_input_scaled = gui_input_scaled
    else:
        gui_input_scaled = (min_value-b)/(max_value-min_value)

    return(gui_input_scaled)

#Creating a Function which takes the number of manual Steps and the Processed Column of Manual Steps and the maximum value of the scaled data and is able to scale the input from User and Return the Scaled Data

def clean_live_data_Manual_Steps(c,column,max_c):
    max_value = column.max()
    min_value = column.min()
    manual_steps_scaled = (c-min_value)/(max_value-min_value)
    if manual_steps_scaled>0:
        if manual_steps_scaled>max_c:
            manual_steps_scaled = max_c
        else:
            manual_steps_scaled = manual_steps_scaled
    else:
        manual_steps_scaled = (min_value-c)/(max_value-min_value)

    return(manual_steps_scaled)

#Creating a Function which takes the number of manual Steps and the Processed Column of Manual Steps and the maximum value of the scaled data and is able to scale the input from User and Return the Scaled Data

def clean_live_data_Transaction_time(d,column,max_d):
    max_value = column.max()
    min_value = column.min()
    time_scaled = (d-min_value)/(max_value-min_value)
    if time_scaled>0:
        if time_scaled>max_d:
            time_scaled = max_d
        else:
            time_scaled = time_scaled
    else:
        time_scaled = (min_value-d)/(max_value-min_value)

    return(time_scaled)

#Creating a Function which takes the number of Volume of Data and the Processed Column of Volume and the maximum value of the scaled data and is able to scale the input from User and Return the Scaled Data

def clean_live_data_Volume(f,column,max_f):
    max_value = column.max()
    min_value = column.min()
    volume_scaled = (f-min_value)/(max_value-min_value)
    if volume_scaled>0:
        if volume_scaled>max_f:
            volume_scaled = max_f
        else:
            volume_scaled = volume_scaled
    else:
        volume_scaled = (min_value-f)/(max_value-min_value)

    return(volume_scaled)

#Creating a Function which takes the number of FTE Covered and the Processed Column of FTE and the maximum value of the scaled data and is able to scale the input from User and Return the Scaled Data

def clean_live_FTE(g,column,max_g):
    max_value = column.max()
    min_value = column.min()
    FTE_scaled = (g-min_value)/(max_value-min_value)
    if FTE_scaled>0:
        if FTE_scaled>max_g:
            FTE_scaled = max_g
        else:
            FTE_scaled = FTE_scaled
    else:
        FTE_scaled = (min_value-g)/(max_value-min_value)

    return(FTE_scaled)

#Creating a Function which considers if there is a Strict SLA or not and the Processed Column of SLA and the maximum value of the scaled data and is able to scale the input from User and Return the Scaled Data

def clean_live_SLA(h,column,max_h):
    max_value = column.max()
    min_value = column.min()
    SLA_scaled = (h-min_value)/(max_value-min_value)
    if SLA_scaled>0:
        if SLA_scaled>max_h:
            SLA_scaled = max_h
        else:
            SLA_scaled = SLA_scaled
    else:
        SLA_scaled = (min_value-h)/(max_value-min_value)

    return(SLA_scaled)

#Creating a Function which considers if there is a Strict SLA or not and the Processed Column of SLA and the maximum value of the scaled data and is able to scale the input from User and Return the Scaled Data

def clean_live_SLA(h,column,max_h):
    max_value = column.max()
    min_value = column.min()
    SLA_scaled = (h-min_value)/(max_value-min_value)
    if SLA_scaled>0:
        if SLA_scaled>max_h:
            SLA_scaled = max_h
        else:
            SLA_scaled = SLA_scaled
    else:
        SLA_scaled = (min_value-h)/(max_value-min_value)

    return(SLA_scaled)

#Creating a Function which considers the number of decision points and the Processed Column of Decision Points and the maximum value of the scaled data and is able to scale the input from User and Return the Scaled Data

def clean_live_Decision_Points(j,column,max_j):
    max_value = column.max()
    min_value = column.min()
    DP_scaled = (j-min_value)/(max_value-min_value)
    if DP_scaled>0:
        if DP_scaled>max_j:
            DP_scaled = max_j
        else:
            DP_scaled = DP_scaled
    else:
        DP_scaled = (min_value-j)/(max_value-min_value)

    return(DP_scaled)

#Creating a Function which considers the number of decision points and the Processed Column of Decision Points and the maximum value of the scaled data and is able to scale the input from User and Return the Scaled Data

def clean_live_data_Business_Exceptions(k,column,max_k):
    max_value = column.max()
    min_value = column.min()
    BE_scaled = (k-min_value)/(max_value-min_value)
    if BE_scaled>0:
        if BE_scaled>max_k:
            BE_scaled = max_k
        else:
            BE_scaled = BE_scaled
    else:
        BE_scaled = (min_value-k)/(max_value-min_value)

    return(BE_scaled)

#Clustering algorithm when FTE is not treated as OverRiding Parameter (Model Is already Trained and Centroids have arrived based on that)

def model_training_with_FTE(final_data):

    print("\n Kindly Wait while the Algorithm is Running")
    print("------------------------------------------------")
    print("\n Model Build Started....")
    
    print("\n Clustering the Data Points with Training Data....")
    
    init=np.array([[1.00, 0.213, 0.082, 0.00076, 0.33, 0.25, 0.539, 0.317],[0.025, 0.0709, 0.096, 0.0085, 0.209, 0, 0.131, 0.0824],[0.0565, 0.217, 0.18, 0.097, 0.259, 1, 0.31, 0.162]],dtype=object)
    kmeans = KMeans(n_clusters=3,init='k-means++',n_init=30,random_state=6)
    kmeans.fit(final_data)
    y_kmeans_random_state = kmeans.predict(final_data)
    
    print("\n Data Clustering is completed...")
    return(kmeans)


#Clustering algorithm when FTE is treated as OverRiding Parameter (Model Is already Trained and Centroids have arrived based on that)

def model_training_without_FTE(final_data):

    print("\n Kindly Wait while the Algorithm is Running")
    print("------------------------------------------------")
    print("\n Model Build Started....")
    
    print("\n Clustering the Data Points with Training Data....")

    init=np.array([[1.00, 0.213, 0.082, 0.00076, 0.33, 0.25, 0.539, 0.317],[0.025, 0.0709, 0.096, 0.0085, 0.209, 0, 0.131, 0.0824],[0.0565, 0.217, 0.18, 0.097, 0.259, 1, 0.31, 0.162]],dtype=object)
    kmeans = KMeans(n_clusters=3,init='k-means++',n_init=30,random_state=6)
    kmeans.fit(final_data)
    y_kmeans_random_state = kmeans.predict(final_data)

    print("\n Data Clustering is completed...")
    return(kmeans)

#Calling a Trained Model of SVM to predict the number of Development days required for Developing Use Case

def development_days_model_training(X_dev,y_dev,b,c,d,h,j,k):

    from sklearn import svm
    input_parameters = [[b,c,d,h,j,k]]
    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev,test_size=0.01,random_state=0)
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(input_parameters)
    #print(y_pred)
    return(y_pred)

#Gathering User Input for Live Data to Test How the Clustering Model is Working

def input_data(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13):

    # print("Please Answer the following questions so that the complexity of the Use Case can be predicted- ")
    # print("--------------------------------------------------------------------------------------------------")
    complex_application_check = int(q1)
    number_of_gui_screens = float(q2)
    number_of_manual_steps = float(q3)
    amount_of_time = float(q4)
    type_of_application = int(q5)
    expected_volume = float(q6)
    expected_fte_workload = float(q7)
    sla_time = int(q8)
    stability_of_application = int(q9)
    business_workflows = float(q10)
    business_exceptions = float(q11)
    integration_required = int(q12)
    complex_methods_used = int(q13)
    # print("--------------------------------------------------------------------------------------------------------")
    return(complex_application_check,number_of_gui_screens,number_of_manual_steps,amount_of_time,type_of_application,expected_volume,expected_fte_workload,sla_time,stability_of_application,business_workflows,business_exceptions,integration_required,complex_methods_used)


#To Check if The Parameters Specified Are OverRiding or not and hence take an action based on that

def check_overriding_parameters(a,e,i,l,m,g):
    parameter_list = np.array([a,e,i,l,m,g])
    paramter_number = [1,5,9,12,13,7]
    parameter_output = []
    
    for i in range(0,6):
        if i<5:
            if parameter_list[i]==1:
                parameter_output.append(paramter_number[i])
            else:
                parameter_output.append(-1)
        else:
            if g>=4:
                parameter_output.append(paramter_number[i])
            else:
                parameter_output.append(-1)

    if any(y > -1 for y in parameter_output):
        flag = True
    else:
        flag = False

    return(parameter_output, flag)

def main(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13):

    #For ignoring the warnings on the console window, putting in the below code
    # warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    warnings.filterwarnings(action='ignore', category=DataConversionWarning)

    #Creating Pandas Dataframes for Fully Scaled Data and Intermediate Data before Scaling but after Data Pre-processing has completed
    transformed_use_case = pd.read_csv('transformed_data.csv')
    scaled_use_case = pd.read_csv('Intermediate_data.csv')
    development_use_case = pd.read_csv("dataset_development_days.csv")
    final_data = transformed_use_case[['GUI_screens','manual_steps','transaction_time','volume','FTE','SLA','Decision_points','Business_exception']]
    X_dev = development_use_case[['GUI_screens','manual_steps','transaction_time','SLA','Decision_points','Business_exception']]
    y_dev = development_use_case[['Development_Days']] 

    #Reading User Input for all the Paramters which can determine the complexity of a use case
    a,b,c,d,e,f,g,h,i,j,k,l,m = input_data(q1,q2,q3,q4,q5,q6,q7,q8,q9,q10,q11,q12,q13)
    
    #Creating Columns for Non-Overriding Parameters so that Scaling can be achieved as per Trained Data
    column_GUI_Screens = (scaled_use_case['GUI_screens'])
    column_manual_steps = (scaled_use_case['manual_steps'])
    column_tt = (scaled_use_case['transaction_time'])
    column_volume = (scaled_use_case['volume'])
    column_FTE = (scaled_use_case['FTE'])
    column_SLA = (scaled_use_case['SLA'])
    column_DP = (scaled_use_case['Decision_points'])
    column_BE = (scaled_use_case['Business_exception'])

    #Figuring Out the Maximum Value in Pre-Processed Data Columns
    max_b= (transformed_use_case['GUI_screens']).max()
    max_c= (transformed_use_case['manual_steps']).max()
    max_d= (transformed_use_case['transaction_time']).max()
    max_f= (transformed_use_case['volume']).max()
    max_g= (transformed_use_case['FTE']).max()
    max_h= (transformed_use_case['SLA']).max()
    max_j= (transformed_use_case['Decision_points']).max()
    max_k= (transformed_use_case['Business_exception']).max()

    #Calling Functions to Scale up the Data which will be sent by Developers while Using the System--- For Doing so passing the actual value with the Trained and Cleaned Column
    scaled_b= clean_live_data_GUI_Screens(b,column_GUI_Screens,max_b)
    scaled_c= clean_live_data_Manual_Steps(c,column_manual_steps,max_c)
    scaled_d= clean_live_data_Transaction_time(d,column_tt,max_d)
    scaled_f= clean_live_data_Volume(f,column_volume,max_f)
    if g<4:
        scaled_g= clean_live_FTE(g,column_FTE,max_g)
    scaled_h= clean_live_SLA(h,column_SLA,max_h)
    scaled_j= clean_live_Decision_Points(j,column_DP,max_j)
    scaled_k= clean_live_data_Business_Exceptions(k,column_BE,max_k)

    #Checking if FTE is an overriding Parameter or Not and If FTE Is considered as a OverRiding Parameter, then Clustering should be happening without it
    if g<4:
        input_test = [[scaled_b,scaled_c,scaled_d,scaled_f,scaled_g,scaled_h,scaled_j,scaled_k]]
        y_kmeans_random_state =  model_training_with_FTE(final_data)
        y_kmeans_pred = y_kmeans_random_state.predict(input_test)
    else:
        final_data_without_FTE = transformed_use_case[['GUI_screens','manual_steps','transaction_time','volume','SLA','Decision_points','Business_exception']]
        input_test = [[scaled_b,scaled_c,scaled_d,scaled_f,scaled_h,scaled_j,scaled_k]]
        y_kmeans_random_state =  model_training_without_FTE(final_data_without_FTE)
        y_kmeans_pred = y_kmeans_random_state.predict(input_test)

    #Support Vector Machine Model which is already trained to predict the value based on development days
    dev_days_predict =0
    weeks_predicted = 0
    dev_days_predict = development_days_model_training(X_dev,y_dev,b,c,d,h,j,k)
    #dev_days_predict = svm_trained(input_parameters)
    weeks_predicted = int(dev_days_predict/5)
    #print(dev_days_predict)

    #Checking if Clustering Algorithm specifies the process to be Simple/Medium/Complex based on Data Points

    process_specific_complexity = ''
    process_specific_message = ''

    if y_kmeans_pred == 0:
        process_specific_message = "\n Process Specific Parameters predict your Use Case as ------> Simple "
        process_specific_complexity ='Simple'
    elif y_kmeans_pred == 1:
        process_specific_message ="\n Process Specific Parameters predict your Use Case as -------> Medium "
        process_specific_complexity ='Medium'
    else:
        process_specific_message ="\n Process Specific Parameters predict your Use Case as -------> Complex "
        process_specific_complexity ='Complex'
    

    #Check If User Input Has OverRiding Parameters and if They are there, check the reason for Overriding
    reason_overriding, overriding_parameter_present = check_overriding_parameters(a,e,i,l,m,g)
    
    #Print onto the console window that why the process has been OverRidden
    if overriding_parameter_present:
        print("\n There are OverRiding Parameters in your Use Case-")
        dev_days_statement =''
        overriding_reason =''
        count=1
        flag_overriding = 0
        
        if any(y == 1 for y in reason_overriding):
            overriding_reason= overriding_reason+str(count)+' The Application used in your Use Case has been categorized as Complex.'+'\n'
            count = count+1
            flag_overriding = 1
            #print("\n OverRiding Parameter for your Use Case- The Application used in your Use Case has been categorized as Complex \n Overall Complexity of your Use Case has been predicted as -----> Complex")
        
        if any(y == 5 for y in reason_overriding):
            overriding_reason = overriding_reason+str(count)+' Your Use Case would require a Remote Access type of Target Application Access'+'\n'
            count = count+1
            flag_overriding = 1
            #print("\n OverRiding Parameter for your Use Case- As your process would require a Remote Access type of Target Application Access, Hence Use Case has been Categorized as Complex")
            #print("Overall Complexity for your Use Case -----> Complex")

        if any(y == 9 for y in reason_overriding):
            overriding_reason = overriding_reason+str(count)+' Stability of the Application used in your Use Case is Low'+'\n'
            count = count+1
            flag_overriding = 1
            #print("\n Overwriting Parameter for your Use Case: As the Stability of the Application used in your Use Case is Low, Hence categorizing Use Case as Complex \n Overall Complexity of your Use Case has been predicted as --- Complex")

        if any(y == 12 for y in reason_overriding):
            overriding_reason = overriding_reason+str(count)+' The Use Case uses or requires some complex Integrations Like NLP, ML, iOCR, Interact'+'\n'
            count = count+1
            flag_overriding = 1
            #print("\n Overwriting Parameter for your Use Case: The Use Case uses or requires some complex Integrations Like NLP, ML, API, iOCR, Interact \n Overall Complexity of your Use Case has been predicted as --- Complex")
    
        if any(y == 7 for y in reason_overriding):
            overriding_reason = overriding_reason+str(count)+' The Use Case saves 4 or Greater than 4 FTE Workload'+'\n'
            count = count+1
            flag_overriding = 1
            #print("\n Overwriting Parameter for your Use Case: The Use Case saves 4 or Greater than 4 FTE Workload \n Overall Complexity of your Use Case has been predicted as --- Complex")
        
        elif (any(y == 13 for y in reason_overriding) and flag_overriding==0):
            overriding_reason=f" OverRiding Parameter for your Use Case: Scripting Needed, Hence OverRiding Parameter Complexity ----- Medium "
            if((y_kmeans_pred==0) or (y_kmeans_pred==1)):
                print("\n Overall Complexity for Use Case is Predicted as Medium")
                dev_days_statement= f"As Scripting would be required for your use case, hence total number of development days is estimated between 30-40 days or 6-8 weeks"
            else:
                print("\n Overall Complexity for your Use Case is predicated as Complex")
                dev_days_statement = f" The Number of Development Days assigned for your use case is close to {weeks_predicted}-{weeks_predicted+1} weeks or close to {int(dev_days_predict)} - {int(dev_days_predict)+5} days"


        if flag_overriding==1:
            overriding_reason = overriding_reason+' Overall Complexity for your Use Case -----> Complex'+'\n'
            print(overriding_reason)
            dev_days_statement = f" The Number of Development Days assigned for your use case is close to {weeks_predicted}-{weeks_predicted+1} weeks or close to {int(dev_days_predict)} - {int(dev_days_predict)+5} days\n However, As there are overriding parameters in your use case, hence total number of development days is estimated between 10-12 weeks"
        
    else:
        dev_days_statement =''
        overriding_reason =''
        if(y_kmeans_pred == 0):
            print("---------------------------------------------------------------------------------------")
            print("\n There are No OverRiding Parameters in your Use Case \n\n Overall Complexity for your Use Case is predicated as Simple")
            if dev_days_predict>15:
                dev_days_statement = f"The Number of Development Days assigned for your use case is close to 2 weeks or 10 days"
            else:
               dev_days_statement = f" The Number of Development Days assigned for your use case is close to {weeks_predicted}-{weeks_predicted+1} weeks or close to {int(dev_days_predict)} - {int(dev_days_predict)+5} days"

        elif(y_kmeans_pred == 1): #y_kmeans_random_state
            print("---------------------------------------------------------------------------------------")
            print("\n There are No OverRiding Parameters in your Use Case \n\n Overall Complexity for your Use Case is predicated as Medium")
            if dev_days_predict>40:
                dev_days_statement=f" The Number of Development Days assigned for your use case is close to 6-7 weeks or 30-35 days"
            else:
                dev_days_statement = f" The Number of Development Days assigned for your use case is close to {weeks_predicted}-{weeks_predicted+1} weeks or close to {int(dev_days_predict)} - {int(dev_days_predict)+5} days"
        else:
            print("---------------------------------------------------------------------------------------")
            print("\n There are No Overriding Parameters in your Use Case \n \n Overall Complexity for your Use Case is predicated as Complex \n")
            if dev_days_predict>60:
                dev_days_statement= f"The Number of Development Days assigned for your use case is close to 9-10 weeks or 45-50 days"
            else:
               dev_days_statement = f" The Number of Development Days assigned for your use case is close to {weeks_predicted}-{weeks_predicted+1} weeks or close to {int(dev_days_predict)} - {int(dev_days_predict)+5} days"
    print("\n ------------------------------------------------------------------------------------------------------------\n")
    return(process_specific_complexity,overriding_parameter_present,overriding_reason,dev_days_statement)

if __name__ == "__main__":
    main()