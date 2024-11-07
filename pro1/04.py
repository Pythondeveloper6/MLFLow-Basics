from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


models = [
    ("LogisticRegressions" , LogisticRegression() , {"max_iter":200}),
    ("DecisionTreeClassifier" , DecisionTreeClassifier() , {"max_depth":5}),
    ("RandomForestClassifier" , RandomForestClassifier() , {"n_estimators":50 , "max_depth":5})
]


for x ,y ,z in models:
    print(x)
    print(y)
    print(z)
    print('------------------')





# for model_name , model , model_params in models:
#     main(model_name,model,model_params,train_size,random_state)