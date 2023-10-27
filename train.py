import sys, os, argparse, random
import logging, threading
from pathlib import Path
import lightgbm as lgb
import numpy as np
import jax.numpy as jnp
import edgeimpulse.jax.lgbm
import edgeimpulse.jax.convert
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys, os, signal, random, time, argparse
import logging, threading

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

    Y_train = np.argmax(Y_train, axis=1)
    Y_test = np.argmax(Y_test, axis=1)

    num_features = MODEL_INPUT_SHAPE[0]
    num_classes = len(input.classes)
    print('num features: ' + str(num_features))
    print('num classes: ' + str(num_classes))

    num_iterations = args.num_iterations or 10
    max_depth = args.max_depth or 20
    print('Num. iterations: ' + str(num_iterations))
    print('Max. depth: ' + str(max_depth))

    if num_classes == 2:
        clf = lgb.LGBMClassifier(num_iterations=num_iterations, max_depth=max_depth, objective='binary')
    else:
        clf = lgb.LGBMClassifier(num_iterations=num_iterations, max_depth=max_depth)

    clf.fit(X_train, Y_train)

    print(' ')
    print('Calculating LGBM random forest accuracy...')
    num_correct = 0
    for idx in range(len(Y_test)):
        pred = clf.predict(X_test[idx].reshape(1, -1))
        if Y_test[idx] == pred[0]:
            num_correct += 1
    print(f'Accuracy (validation set): {num_correct / len(Y_test)}')

    clf = clf.booster_

    file_float32 = os.path.join(args.out_directory, 'model.tflite')

    try:
        print('Converting to TensorFlow Lite...')
        if num_classes == 2:
            lgbm = edgeimpulse.jax.lgbm.LGBM(clf, [1, num_features], 1)
        else:
            lgbm = edgeimpulse.jax.lgbm.LGBM(clf, [1, num_features], num_classes)

        def pred_jax(x):
            return lgbm.predict(x)

        edgeimpulse.jax.convert.convert(fun=pred_jax,
                                        model_path=file_float32,
                                        lgbm=lgbm,
                                        input_signature=[
                                            tf.TensorSpec(shape=[1, num_features], dtype=tf.float32, name='input')
                                        ],
                                        output_signature=[1, 1])
        print('Converting to TensorFlow Lite OK')
        print('')
    except Exception as e:
        if (os.path.exists(file_float32)):
            os.unlink(file_float32)
        print('WARN: Failed to convert to TensorFlow Lite:', e)
        print('')
        exit(1)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    main_function()