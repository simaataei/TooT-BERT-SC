from Substrate_classifier import SubstrateClassifier
import optuna
from sklearn.model_selection import train_test_split
from Data_prepration import read_data

def objective(trial):
    # define the search space for hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
    num_epochs = trial.suggest_int('num_epochs', 10, 100)
    batch_size = trial.suggest_int('batch_size', 8, 32)
    early_stopping = trial.suggest_int('early_stopping', 5, 20)

    # create an instance of the SubstrateClassifier class
    model = SubstrateClassifier(num_classes=10, dropout_rate=0.1, learning_rate=learning_rate)

    # train the model and get the validation loss of the last epoch
    val_loss = model.trainer(num_epochs=num_epochs, batch_size=batch_size, train_set=list(zip(X_train, y_train)), val_set=list(zip(X_val, y_val)), early_stopping=early_stopping, learning_rate=learning_rate)

    # return the validation loss of the last epoch
    return val_loss[-1]




X, y = read_data()

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print('Best hyperparameters: ', study.best_params)
best_model = SubstrateClassifier(num_classes=2, batch_size=study.best_params['batch_size'])
best_model.trainer(num_epochs=study.best_params['num_epochs'], batch_size=study.best_params['batch_size'], train_set=list(zip(X_train_val, y_train_val)), val_set=list(zip(X_test, y_test)), early_stopping=study.best_params['early_stopping'], learning_rate=study.best_params['learning_rate'])
best_model.test(list(zip(X_test, y_test)))
