import os
import pickle
import shutil
import tensorflow as tf

from utils.losses import *


class ModelStorage:
    training_filename = 'training_{type}'
    val_filename = 'val_{type}'
    test_filename = 'test_{type}'

    training_path = '/train'
    val_path = '/val'
    test_path = '/test'
    checkpoint_path = '/model_checkpoint'
    logs_path = '/logs'

    def __init__(self, path):
        self._path = path

        self.create_dirs(self._path + self.checkpoint_path)
        self.create_dirs(self._path + self.logs_path)
        self.create_dirs(self._path + self.training_path)
        self.create_dirs(self._path + self.val_path)
        self.create_dirs(self._path + self.test_path)

    def create_dirs(self, fullpath, remove_existing=False):
        if remove_existing and os.path.exists(fullpath):
            shutil.rmtree(fullpath)
        if not os.path.exists(fullpath):
            os.makedirs(fullpath)

    def _save_keras_model(self, model, epoch):
        if epoch >= 0:
            model.save(self._path + self.checkpoint_path + '/model_' +
                       str(epoch) + '.h5')
        else:
            model.save(self._path + self.checkpoint_path + '/model.h5')

    def _load_keras_model(self, path, args):
        return tf.keras.models.load_model(path, custom_objects={
            'Y': np.empty((0, 2)),
            'multi_dim_gaussian_nll': multi_dim_gaussian_nll,
            'gaussian_nll': gaussian_nll, 'gaussian_mse': gaussian_mse, 'gaussian_mae': gaussian_mae})

    def save_model(self, model, model_backend, args, epoch=-1):
        if not epoch % args.dump == 0:
            return

        if model_backend == 'keras':
            self._save_keras_model(model, epoch)

    def load_model(self, path, model_backend, args):
        if model_backend == 'keras':
            return self._load_keras_model(path, args)

    def save_sets(self, train, val, test):
        self.create_dirs(self._path + self.training_path, True)
        self.create_dirs(self._path + self.val_path, True)
        self.create_dirs(self._path + self.test_path, True)

        with open(self._path + self.training_path + '/' + self.training_filename.replace('{type}', 'inputs') + '.pkl', 'wb') as f:
            pickle.dump(train[0], f)

        with open(self._path + self.training_path + '/' + self.training_filename.replace('{type}', 'ouputs') + '.pkl', 'wb') as f:
            pickle.dump(train[1], f)

        with open(self._path + self.val_path + '/' + self.val_filename.replace('{type}', 'inputs') + '.pkl', 'wb') as f:
            pickle.dump(val[0], f)

        with open(self._path + self.val_path + '/' + self.val_filename.replace('{type}', 'ouputs') + '.pkl', 'wb') as f:
            pickle.dump(val[1], f)

        with open(self._path + self.test_path + '/' + self.test_filename.replace('{type}', 'inputs') + '.pkl', 'wb') as f:
            pickle.dump(test[0], f)

        with open(self._path + self.test_path + '/' + self.test_filename.replace('{type}', 'ouputs') + '.pkl', 'wb') as f:
            pickle.dump(test[1], f)

    def get_training_path(self):
        return self._path + self.training_path

    def get_val_path(self):
        return self._path + self.val_path

    def get_test_path(self):
        return self._path + self.test_path

    def get_checkpoint_path(self):
        return self._path + self.checkpoint_path

    def get_logs_path(self):
        return self._path + self.logs_path
