import mlflow

# create new expr
new_exper = mlflow.create_experiment(
    name='test2' , # project name 
    artifact_location= 'test2_artifacts', # save location for data
    tags={"env":"dev" , "version":"0.1","team":"Mahmoud"}
)


# i want to use expr
mlflow.set_experiment("test2")


# save value : n_estimators = 30
mlflow.log_param("n_estimators",30)
mlflow.log_metric