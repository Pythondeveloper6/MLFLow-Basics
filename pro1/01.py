from sklearn.datasets import load_diabetes  # dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
    
    # load data 
    data = load_diabetes()
    x = data.data
    y = data.target

    # split data 
    x_train , x_test , y_train , y_test = train_test_split(
        x , y , test_size=.2 , random_state=42
    )

    # build model
    model = LinearRegression()
    model.fit(x_train,y_train)

    # male prediction
    y_pred = model.predict(x_test)

    # calculate accuracy
    score = r2_score(y_test,y_pred)
    print(f"Score : {score}")



if __name__ == "__main__":
    main()