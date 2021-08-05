# Save Model Using joblib
import joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression


def main():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    names = ['preg', 'plas', 'pres', 'skin',
             'test', 'mass', 'pedi', 'age', 'class']
    dataframe = pandas.read_csv(url, names=names)
    array = dataframe.values
    X = array[:, 0:8]
    Y = array[:, 8]
    test_size = 0.33
    seed = 7
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y, test_size=test_size, random_state=seed)
    # Fit the model on training set
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # save the model to disk
    filename = 'ml/model/model.pkl'
    joblib.dump(model, filename)
    # Save a test record for postman
    # test_pred =


if __name__ == "__main__":
    main()
