import tensorflow as tf


# Define functions for loading data.
# Turn a loaded JPEG image into a tensor.


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
def load_train_dataset(filenames, labels, image_size):
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
        metrics=metrics
    )

    return model
