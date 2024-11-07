import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes  # dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    # create expre
    # mlflow.create_experiment(
    #     name="diabetes_regression",
    #     artifact_location = "diabetes_regression_artifacts",
    #     tags={"version":'01',"team":"Mahmoud"}
    # )

    # use exper
    mlflow.set_experiment("diabetes_regression")

    # param
    train_size = 0.1
    random_state = 10

    # load data 
    data = load_diabetes()
    x = data.data
    y = data.target

    # split data 
    x_train , x_test , y_train , y_test = train_test_split(
        x , y , test_size=train_size , random_state=random_state
    )

    with mlflow.start_run(run_name='test3'):   # start mlflow track 
        # log params
        mlflow.log_param("test_size",train_size)
        mlflow.log_param("random_state",random_state)

        # build model
        model = LinearRegression()
        model.fit(x_train,y_train)

        # male prediction
        y_pred = model.predict(x_test)

        # calculate accuracy
        score = r2_score(y_test,y_pred)
        print(f"Score : {score}")

        # save score
        mlflow.log_metric("r2_score",score)

        # save model
        mlflow.sklearn.log_model(model,"LinearRegression")



if __name__ == "__main__":
    main()