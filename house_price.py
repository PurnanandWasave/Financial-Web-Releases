import numpy as np
from sklearn import datasets
from sklearn.svm import SVM
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

#inputting
data = datasets.load_boston()

X, y = shuffle(data.data, data.target, random_state=7)

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]


#Support Vector Regression Model
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)

#Training the model
sv_regressor.fit(X_train, y_train)


#Evaluation 
y_test-pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print("\n#####Performance#####")
print("Mean squared error =", round(mse, 2))
print("Explained Variance Score =", round(evs, 2))

test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]
print("\nPredicted price :", sv_regressor.predict([test_data])[0])


####House Price Predictor is ready####






















