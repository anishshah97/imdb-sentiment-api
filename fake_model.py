# Save Model Using joblib
from pathlib import Path

import joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

import wandb
from wandb import Artifact

run = wandb.init(project="imdb-sentiment-api", job_type="training")


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
    # # Fit the model on training set
    trained_model_artifact = Artifact(
        "model", type="model", description="test model for fastapi wandb cd flow")
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # # save the model to disk
    model_dir = Path("ml", "model")
    file_path = Path(model_dir, "model.pkl")
    joblib.dump(model, file_path)
    trained_model_artifact.add_file(file_path)
    test_preds = model.predict(X_test)
    wandb.sklearn.plot_confusion_matrix(Y_test, test_preds, Y)
    run.log_artifact(trained_model_artifact)


if __name__ == "__main__":
    main()
