import sys, os, argparse
import lightgbm as lgb
import numpy as np
import sys, os, signal, random, time, argparse
from sklearn import metrics


RANDOM_SEED = 1
dir_path = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='Edge Impulse LightGBM training scripts')

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

print('')
print('Training LGBM model...')

print('mode: regression')

dtrain = lgb.Dataset(X_train, label=Y_train)
dval = lgb.Dataset(X_test, label=Y_test)

params = {
    "objective": "regression",
    "num_classes": 1
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

predicted_y = clf.predict(X_test)
print('r^2: ' + str(metrics.r2_score(Y_test, predicted_y)))
print('mse: ' + str(metrics.mean_squared_error(Y_test, predicted_y)))
print('log(mse): ' + str(metrics.mean_squared_log_error(Y_test, predicted_y)))

print('')
print('Saving LGBM model...')
file_lgbm = os.path.join(args.out_directory, 'model.txt')
clf.save_model(file_lgbm)
print('Saving LGBM model OK')
