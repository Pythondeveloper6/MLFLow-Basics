import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def main(model_name,model,model_params,test_size , random_state):
    # get data
    data = load_iris()
    x = data.data
    y = data.target

    # split data 
    x_train , x_test , y_train , y_test = train_test_split(
        x , y , test_size=test_size , random_state=random_state
    )

    with mlflow.start_run(run_name=model_name):
        # log params 
        mlflow.log_param("test_size",test_size)
        mlflow.log_param("random_state",random_state)

        # loop over each model params 
        for param_name , param_value in model_params.items():
            mlflow.log_param(param_name,param_value)
        
        # build model
        model.set_params(**model_params)   # LogisticRegression(max_iter=200) {"max_iter":200}
        model.fit(x_train,y_train)

        # make prediction
        y_pred = model.predict(x_test)
        
        # accuracy
        accuracy = accuracy_score(y_test,y_pred)

        # save in mlflow
        mlflow.log_metric("accuracy",accuracy)
        mlflow.sklearn.log_model(model,model_name)


if __name__ == "__main__":    
    # use expr
    mlflow.set_experiment("iris_classification")

    models = [
        ("LogisticRegressions3" , LogisticRegression() , {"max_iter":100}),
        ("DecisionTreeClassifier3" , DecisionTreeClassifier() , {"max_depth":3}),
        ("RandomForestClassifier3" , RandomForestClassifier() , {"n_estimators":30 , "max_depth":7})
    ]
    # param
    train_size = 0.15
    random_state = 15

    for model_name , model , model_params in models:
        main(model_name,model,model_params,train_size,random_state)