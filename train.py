import sys, os, argparse
import lightgbm as lgb
import numpy as np
import sys, os, signal, random, time, argparse
import logging, threading
from sklearn import metrics

sys.path.append('./resources/libraries')
import ei_tensorflow.training
from ei_shared.parse_train_input import parse_train_input, parse_input_shape


RANDOM_SEED = 1
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Edge Impulse training scripts')
parser.add_argument('--info-file', type=str, required=False,
                    help='train_input.json file with info about classes and input shape',
                    default=os.path.join(dir_path, 'train_input.json'))
parser.add_argument('--data-directory', type=str, required=True,
                    help='Where to read the data from')
parser.add_argument('--out-directory', type=str, required=True,
                    help='Where to write the data')

parser.add_argument('--learning-rate', type=float, required=False,
                    help='Step size shrinkage used in update to prevents overfitting.')
parser.add_argument('--num-iterations', type=int, required=False,
                    help='Number of training cycles')
parser.add_argument('--max-depth', type=int, required=False,
                    help='Maximum depth')

args, unknown = parser.parse_known_args()

# Info about the training pipeline (inputs / shapes / modes etc.)
if not os.path.exists(args.info_file):
    print('Info file', args.info_file, 'does not exist')
    exit(1)

input = parse_train_input(args.info_file)

BEST_MODEL_PATH = os.path.join(os.sep, 'tmp', 'best_model.tf' if input.akidaModel else 'best_model.hdf5')

# Information about the data and input:
# The shape of the model's input (which may be different from the shape of the data)
MODEL_INPUT_SHAPE = parse_input_shape(input.inputShapeString)
# The length of the model's input, used to determine the reshape inside the model
MODEL_INPUT_LENGTH = MODEL_INPUT_SHAPE[0]
MAX_TRAINING_TIME_S = input.maxTrainingTimeSeconds


def exit_gracefully(signum, frame):
    print("")
    print("Terminated by user", flush=True)
    time.sleep(0.2)
    sys.exit(1)

def main_function():
    """This function is used to avoid contaminating the global scope"""

    train_dataset, validation_dataset, samples_dataset, X_train, X_test, Y_train, Y_test, has_samples, X_samples, Y_samples = ei_tensorflow.training.get_dataset_from_folder(
        input, args.data_directory, RANDOM_SEED, None, MODEL_INPUT_SHAPE, None
    )

    print('')
    print('Training LGBM model...')

    if input.mode == 'classification':
        Y_train = np.argmax(Y_train, axis=1)
        Y_test = np.argmax(Y_test, axis=1)

    num_iterations = args.num_iterations or 10
    max_depth = args.max_depth or 20
    num_features = MODEL_INPUT_SHAPE[0]
    num_classes = len(input.classes)
    learning_rate = args.learning_rate or 0.3

    print('Num. iterations: ' + str(num_iterations))
    print('Max. depth: ' + str(max_depth))
    print('Learning rate: ' + str(learning_rate))
    print('num features: ' + str(num_features))
    print('num classes: ' + str(num_classes))
    print('mode: ' + str(input.mode))

    dtrain = lgb.Dataset(X_train, label=Y_train)
    dval = lgb.Dataset(X_test, label=Y_test)

    params = None
    if input.mode == 'regression':
        params = {
            "objective": "regression",
            "num_classes": 1
        }
    else:
        if num_classes == 2:
            params = {
                "objective": "binary",
                "num_classes": 1
            }
        else:
            params = {
                "objective": "multiclass",
                "num_classes": num_classes
            }

    params['learning_rate'] = learning_rate

    clf = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        callbacks=[lgb.log_evaluation()]
    )

    print(' ')
    print('Calculating LGBM random forest accuracy...')

    if input.mode == 'regression':
        predicted_y = clf.predict(X_test)
        print('r^2: ' + str(metrics.r2_score(Y_test, predicted_y)))
        print('mse: ' + str(metrics.mean_squared_error(Y_test, predicted_y)))
        print('log(mse): ' + str(metrics.mean_squared_log_error(Y_test, predicted_y)))
    else:
        num_correct = 0
        for idx in range(len(Y_test)):
            pred = clf.predict(X_test[idx].reshape(1, -1))
            if num_classes == 2:
                if Y_test[idx] == int(round(pred[0])):
                    num_correct += 1
            else:
                pred = np.argmax(pred, axis=1)
                if Y_test[idx] == pred[0]:
                    num_correct += 1
        print(f'Accuracy (validation set): {num_correct / len(Y_test)}')

    print('Saving LGBM model...')
    file_lgbm = os.path.join(args.out_directory, 'model.txt')
    clf.save_model(file_lgbm)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    main_function()