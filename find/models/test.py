#!/usr/bin/env python

import pickle
import argparse
import numpy as np
from glob import glob
import functools

from find.models.storage import ModelStorage

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate test results')
    parser.add_argument('--path', '-p', type=str,
                        help='Path to the experiment',
                        required=True)
    parser.add_argument('--backend',
                        help='Backend selection',
                        default='keras',
                        choices=['keras', 'trajnet'])
    args = parser.parse_args()

    print('Using {} backend'.format(args.backend))
    if args.backend == 'keras':
        from tensorflow.keras.callbacks import CSVLogger
        models = glob('{}/model_checkpoint/model_*.h5'.format(args.path))
        model_storage = ModelStorage(args.path)

        def compare(item1, item2):
            if int(item1.split('_')[-1].split('.')[0]) < int(item2.split('_')[-1].split('.')[0]):
                return -1
            elif int(item1.split('_')[-1].split('.')[0]) > int(item2.split('_')[-1].split('.')[0]):
                return 1
            else: 
                return 0

        models.sort(key=functools.cmp_to_key(compare))

        epochs = []
        measurements = []
        mnames = ''
        for model_path in models:
            model = model_storage.load_model(
                    model_path, args.backend, args)

            f = open('{}/test/test_inputs.pkl'.format(args.path),'rb')
            X_test = pickle.load(f)
            f.close()
            
            f = open('{}/test/test_outputs.pkl'.format(args.path),'rb')
            Y_test = pickle.load(f)
            f.close()

            h = model.evaluate(x=X_test, y=Y_test,verbose=False)
            epochs.append(int(model_path.split('_')[-1].split('.')[0]))
            measurements.append(h)
            mnames = model.metrics_names
        measurements = np.array(measurements)

        f = open('{}/test/history.csv'.format(args.path), 'w')
        header = 'epochs,' + ','.join(str(x) for x in mnames)
        f.write('{}\n'.format(header))
        for e, data in enumerate(measurements):
            row_str = ','.join(str(x) for x in data)
            f.write('{},{}\n'.format(epochs[e], row_str))
        f.close()