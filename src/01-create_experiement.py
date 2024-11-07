import mlflow


new_exper = mlflow.create_experiment(
    name='heart_attack_prediction' , # project name 
    artifact_location= 'heart_attack_prediction_artifacts', # save location for data
    tags={"env":"dev" , "version":"0.1","team":"Mahmoud"}
)

print(new_exper)