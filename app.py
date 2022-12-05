from flask import Flask, render_template,request
from training_dataset_algorithm import  main

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    complex_application_check = (request.form.get("q1"))
    
    number_of_gui_screens = (request.form.get("q2"))
    
    number_of_manual_steps = (request.form.get("q3"))
    amount_of_time = (request.form.get("q4"))
    type_of_application = (request.form.get("q5"))
    expected_volume = (request.form.get("q6"))
    expected_fte_workload = (request.form.get("q7"))
    sla_time = (request.form.get("q8"))
    stability_of_application = (request.form.get("q9"))
    business_workflows = (request.form.get("q10"))
    business_exceptions = (request.form.get("q11"))
    integration_required = (request.form.get("q12"))
    complex_methods_used = (request.form.get("q13"))
    
    process_specific_complexity,overriding_parameter_present,overriding_reason,dev_days_statement=main(complex_application_check,number_of_gui_screens,number_of_manual_steps,amount_of_time,type_of_application,expected_volume,expected_fte_workload,sla_time,stability_of_application,business_workflows,business_exceptions,integration_required,complex_methods_used)
    
    
    if(overriding_parameter_present):
        
        overriding_parameter_text = "There are OverRiding Parameters in your Use Case"
        return render_template('predict.html',val = [process_specific_complexity,overriding_parameter_text,overriding_reason,dev_days_statement])
    else:
        
        overriding_parameter_text = f"There are No OverRiding Parameters in your Use Case"
        return render_template('predict.html',val = [process_specific_complexity,overriding_parameter_text,overriding_reason,dev_days_statement])

