import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse
import os
import numpy as np
import pandas as pd
import json


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    # What's the difference b/w reshape and resize?
    image = tf.image.resize(image, IMAGE_SIZE)
    return image


# Load JPEG files.
def load_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# Make a dataset.
def load_train_dataset(filenames, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(filenames)
    image_ds = path_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)  # Load labels.
    # Zip images and labels.
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    num_images = len(filenames)

    ds_out = image_label_ds.apply(
        tf.data.experimental.shuffle_and_repeat(buffer_size=num_images)
    )
    ds_out = ds_out.batch(BATCH_SIZE)
    ds_out = ds_out.prefetch(buffer_size=AUTOTUNE)

    return ds_out, num_images  # Return a dataset and number of items.


def load_valid_dataset(filenames, labels):
    path_ds = tf.data.Dataset.from_tensor_slices(filenames)
    image_ds = path_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)  # Load labels.
    # Zip images and labels.
    image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))

    num_images = len(filenames)

    ds_out = image_label_ds.batch(BATCH_SIZE)
    ds_out = ds_out.cache()
    ds_out = ds_out.prefetch(buffer_size=AUTOTUNE)

    return ds_out, num_images  # Return a dataset and number of items.


def load_test_dataset(filenames):
    path_ds = tf.data.Dataset.from_tensor_slices(filenames)
    image_ds = path_ds.map(load_preprocess_image, num_parallel_calls=AUTOTUNE)

    ds_out = image_ds.batch(BATCH_SIZE)
    ds_out = ds_out.prefetch(buffer_size=AUTOTUNE)

    return ds_out  # Return an image dataset alone.


def build_lrfn(
    lr_start=0.00001, lr_max=0.000075,
    lr_min=0.000001, lr_rampup_epochs=20,
    lr_sustain_epochs=0, lr_exp_decay=.8
):
    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay ** (epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    return lrfn


def make_model(output_bias = None, metrics = None):
    # Create the base model from the pre-trained model MobileNet V2

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    base_model = tf.keras.applications.VGG16(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(
            8, activation='relu'
        ),
        tf.keras.layers.Dense(
            1, activation='sigmoid',
            bias_initializer=output_bias
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(),
        loss='binary_crossentropy',
        metrics=[metrics]
    )

    return model


def train(train_ds, valid_ds):
    # Make and train a model.
    model = make_model(
        metrics=tf.keras.metrics.AUC(name='auc')
    )

    lrfn = build_lrfn(lr_max=args.lr_max)
    STEPS_PER_EPOCH = num_train_images // args.batch_size
    VALID_STEPS = num_valid_images // args.batch_size

    model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_ds,
        callbacks=[tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)]
    )

    results = model.evaluate(x_test, y_test)
    print('test loss, test auc:', results)
    
    return model

    
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
    BATCH_SIZE = args.batch_size

 
    # Load images and split them into train and validation sets.
    path_to_images = tf.io.gfile.glob(os.path.join(args.train, '*.jpg'))
    labels = pd.read_csv(os.path.join(args.train, 'retained_labels.csv'))
    path_to_images.sort()  # Sort both path names and labels data frame so that they have the same order.
    labels.sort_values(by=['id'], inplace=True)
    
    train_fnames, valid_fnames, train_labels, valid_labels = train_test_split(
        path_to_images,
        labels['label'].values,
        test_size=0.2,
        random_state=0
    )

    # Make datasets.
    train_ds, num_train_images = load_train_dataset(train_fnames, train_labels)
    valid_ds, num_valid_images = load_valid_dataset(valid_fnames, valid_labels)
    

    # Make and train a model.
    model = train(train_ds, valid_ds)

    # save model to an S3 directory with version number '00000001'
    model.save(os.path.join(args.sm_model_dir, '000000001'), 'vgg16_model.h5')
