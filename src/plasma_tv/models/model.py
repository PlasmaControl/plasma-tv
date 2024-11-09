import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score

# Define the LinearRegression model
def train_linear_regression(X_train, y_train):
    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)
    return clf

# Define the Ridge model
def train_ridge(X_train, y_train, alpha=1.0):
    clf = linear_model.Ridge(alpha=alpha)
    clf.fit(X_train, y_train)
    return clf

# Define the Lasso model
def train_lasso(X_train, y_train, alpha=1.0):
    clf = linear_model.Lasso(alpha=alpha)
    clf.fit(X_train, y_train)
    return clf

# Evaluate sci-kit learn model
def evaluate_sklearn_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Example usage
if __name__ == "__main__":
    # Assuming numpy arrays for sklearn are defined elsewhere:
    # train_loader, test_loader, X_train, y_train, X_test, y_test
    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(10, 10)
    y_test = np.random.rand(10)
    
    # Scikit-Learn Model Training and Evaluation
    sklearn_model = train_linear_regression(X_train, y_train)
    sklearn_accuracy = evaluate_sklearn_model(sklearn_model, X_test, y_test)
    print(f"Scikit-Learn Model Accuracy: {sklearn_accuracy}")