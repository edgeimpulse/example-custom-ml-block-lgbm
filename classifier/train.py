import sys, os, argparse
import lightgbm as lgb
import numpy as np
import sys, os, signal, random, time, argparse
from sklearn import metrics


RANDOM_SEED = 1
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Edge Impulse LightGBM training scripts')
parser.add_argument('--info-file', type=str, required=False,
                    help='train_input.json file with info about classes and input shape',
                    default=os.path.join(dir_path, 'train_input.json'))
parser.add_argument('--data-directory', type=str, required=True,
                    help='Where to read the data from')
parser.add_argument('--out-directory', type=str, required=True,
                    help='Where to write the data')

parser.add_argument('--learning-rate', type=float, required=False, default=0.1,
                    help='Step size shrinkage used in update to prevents overfitting.')
parser.add_argument('--num-iterations', type=int, required=False, default=10,
                    help='Number of training cycles')
parser.add_argument('--max-depth', type=int, required=False, default=-1,
                    help='Maximum depth')

parser.add_argument('--l2', type=float, required=False, default=0.0,
                    help='L2 regularization term on weights. Increasing this value will make model more conservative.')
parser.add_argument('--l1', type=float, required=False, default=0.0,
                    help='L1 regularization term on weights. Increasing this value will make model more conservative.')
parser.add_argument('--max-leaves', type=int, required=False,
                    help='Maximum number of nodes to be added.')
parser.add_argument('--min-child-weight', type=float, required=False, default=0.001,
                    help='Minimum sum of instance weight (hessian) needed in a child.')
parser.add_argument('--feature-fraction', type=float, required=False, default=1.0,
                    help='Subsample ratio of the training instances.')

args, unknown = parser.parse_known_args()

out_directory = args.out_directory

if not os.path.exists(out_directory):
    os.mkdir(out_directory)

# grab train/test set
X_train = np.load(os.path.join(args.data_directory, "X_split_train.npy"))
Y_train = np.load(os.path.join(args.data_directory, "Y_split_train.npy"))
X_test = np.load(os.path.join(args.data_directory, "X_split_test.npy"))
Y_test = np.load(os.path.join(args.data_directory, "Y_split_test.npy"))

num_classes = Y_train.shape[1]

# sparse representation of the labels
Y_train = np.argmax(Y_train, axis=1)
Y_test = np.argmax(Y_test, axis=1)

print('')
print('Training LGBM model...')

print('num classes: ' + str(num_classes))
print('mode: classifier')

dtrain = lgb.Dataset(X_train, label=Y_train)
dval = lgb.Dataset(X_test, label=Y_test)

params = None
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

params['learning_rate'] = args.learning_rate
params['max_depth'] = args.max_depth
params['lambda_l2'] = args.l2
params['lambda_l1'] = args.l1
params['min_child_weight'] = args.min_child_weight
params['feature_fraction'] = args.feature_fraction
params['num_iterations'] = args.num_iterations

print('params:', params)
print('')
print('Training LGBM random forest...')

clf = lgb.train(
    params,
    dtrain,
    valid_sets=[dtrain, dval],
    callbacks=[lgb.log_evaluation()]
)

print('Training LGBM random forest OK')
print('')
print('Calculating LGBM random forest accuracy...')

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

print('')
print('Saving LGBM model...')
file_lgbm = os.path.join(args.out_directory, 'model.txt')
clf.save_model(file_lgbm)
print('Saving LGBM model OK')
