# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.import tensorflow as tf

import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
import os
import numpy as np
import pandas as pd
import json


def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--sm_model_dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--hosts', type=list,
                        default=json.loads(os.environ.get('SM_HOSTS')))
    parser.add_argument('--current_host', type=str,
                        default=os.environ.get('SM_CURRENT_HOST'))

    parser.add_argument('--lr_max', type=float, default=0.000075)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=1024)


    return parser.parse_known_args()


if __name__ == "__main__":
    args, unknown = _parse_args()
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMAGE_SIZE = [args.image_size, args.image_size]

    print("Num GPUs Available: ", len(
        tf.config.experimental.list_physical_devices('GPU')))

    # Load images and split them into train and validation sets.
    train_fnames, valid_fnames, train_labels, valid_labels = train_test_split(
        tf.io.gfile.glob(os.path.join(args.train, '*.jpg')),
        np.load(os.path.join(args.train, 'labels_retained.npy')),
        test_size=0.2,
        random_state=0
    )

    # Make datasets.
    train_ds, num_train_images = load_train_dataset(train_fnames, train_labels)
    valid_ds, num_valid_images = load_valid_dataset(valid_fnames, valid_labels)


    # Make and train a model.
    model = make_model(
        metrics=tf.keras.metrics.AUC(name='auc')
    )

    lrfn = build_lrfn(lr_max=args.lr_max)
    STEPS_PER_EPOCH = num_train_images // args.batch_size
    VALID_STEPS = num_valid_images // args.batch_size

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_ds,
        validation_steps=VALID_STEPS,
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
        ]
    )

    # save model to an S3 directory with version number '00000001'
    model.save(os.path.join(args.sm_model_dir, '000000001'), 'vgg16_model.h5')
