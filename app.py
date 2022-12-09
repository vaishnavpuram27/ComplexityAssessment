from flask import Flask, render_template,request
from training_dataset_algorithm import  main

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    complex_application_check = (request.form.get("q1"))
    if complex_application_check == "True":
        complex_application_check_val=1
    else:
        complex_application_check_val=0
    
    number_of_gui_screens = (request.form.get("q2"))
    
    number_of_manual_steps = (request.form.get("q3"))
    amount_of_time = (request.form.get("q4"))
    type_of_application = (request.form.get("q5"))
    print(type_of_application)
    if type_of_application == "Direct Access":
        type_of_application_val=0
    else:
        type_of_application_val=1
    expected_volume = (request.form.get("q6"))
    expected_fte_workload = (request.form.get("q7"))
    sla_time = (request.form.get("q8"))
    if sla_time == "True":
        sla_time_val=1
    else:
        sla_time_val=0
    stability_of_application = (request.form.get("q9"))
    if stability_of_application == "Low":
        stability_of_application_val=1
    else:
        stability_of_application_val=0
    business_workflows = (request.form.get("q10"))
    business_exceptions = (request.form.get("q11"))
    integration_required = (request.form.get("q12"))
    if integration_required == "True":
        integration_required_val=1
    else:
        integration_required_val=0
    complex_methods_used = (request.form.get("q13"))
    if complex_methods_used == "True":
        complex_methods_used_val=1
    else:
        complex_methods_used_val=0
    useCaseName = (request.form.get("UseCaseName"))
    bizStream = (request.form.get("BizStream"))
    bizAnalyst = (request.form.get("BizAnalyst"))
    seniorDev = (request.form.get("SeniorDev"))
    #useCaseName,bizStream,bizAnalyst,seniorDev
    process_specific_complexity,overriding_parameter_present,overriding_reason,dev_days_statement=main(complex_application_check_val,number_of_gui_screens,number_of_manual_steps,amount_of_time,type_of_application_val,expected_volume,expected_fte_workload,sla_time_val,stability_of_application_val,business_workflows,business_exceptions,integration_required_val,complex_methods_used_val)
    
    
    if(overriding_parameter_present):
        
        overriding_parameter_text = "There are OverRiding Parameters in your Use Case"
        return render_template('predict.html',val = [complex_application_check,number_of_gui_screens,number_of_manual_steps,amount_of_time,type_of_application,expected_volume,expected_fte_workload,sla_time,stability_of_application,business_workflows,business_exceptions,integration_required,complex_methods_used,process_specific_complexity,overriding_parameter_text,overriding_reason,dev_days_statement,useCaseName,bizStream,bizAnalyst,seniorDev])
    else:
        
        overriding_parameter_text = f"There are No OverRiding Parameters in your Use Case"
        return render_template('predict.html',val = [complex_application_check,number_of_gui_screens,number_of_manual_steps,amount_of_time,type_of_application,expected_volume,expected_fte_workload,sla_time,stability_of_application,business_workflows,business_exceptions,integration_required,complex_methods_used,process_specific_complexity,overriding_parameter_text,overriding_reason,dev_days_statement,useCaseName,bizStream,bizAnalyst,seniorDev])

