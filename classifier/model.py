import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras import regularizers

def data_loader(image_size, directory):
    train =tf.keras.preprocessing.image_dataset_from_directory(
    directory=directory, labels='inferred', label_mode='int',
    batch_size=32, image_size=image_size, seed=12, validation_split=0.2, subset="training")

    validate=tf.keras.preprocessing.image_dataset_from_directory(
    directory='C:/Users/Maciek/Desktop/cells/train', labels='inferred', label_mode='int',
    batch_size=32, image_size=image_size, seed=12, validation_split=0.2, subset="validation")

    class_names = train.class_names

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train =train.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validate = validate.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)
    return train, validate, class_names, num_classes

def sequential(num_classes):
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)])

    model = tf.keras.Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.5),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dense(num_classes)
        ])
    return model

def train_seq(model, train, validate, model_name):
    early=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, mode='min', verbose=1, patience=10, restore_best_weights=True)
    learning_rate=0.01 

    model.compile(optimizer=tf.keras.optimizers.SGD(lr=learning_rate, nesterov=True),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    epochs=300

    checkpoint_filepath = 'logs/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_freq=1)

    history = model.fit(
        train,
        validation_data=validate,
        epochs=epochs,
        callbacks=[model_checkpoint_callback, early]
        )

    #acc = history.history['accuracy']
    #val_acc = history.history['val_accuracy']

    #loss = history.history['loss']
    #val_loss = history.history['val_loss']

    #epochs_range = range((early.stopped_epoch+1))

    model.load_weights(checkpoint_filepath)
    model.save(model_name)
    return history, early


def plot(history, early):
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range((early.stopped_epoch+1))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def transfer_learning(train, num_classes, image_size):
    data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", input_shape=(160, 160,3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1)])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.Rescaling(1./127.5, offset= -1)

    IMG_SHAPE = image_size + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')

    image_batch, label_batch = next(iter(train))
    feature_batch = base_model(image_batch)
    #print(feature_batch.shape)

    base_model.trainable = False
    #base_model.summary()

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    #print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(num_classes)
    prediction_batch = prediction_layer(feature_batch_average)
    #print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(160, 160 , 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def train_tuning(train, validate, model, base_model, model_name):
    early=tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, mode='min', verbose=1, patience=10, restore_best_weights=True)

    checkpoint_filepath = 'logs/checkpoint1'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_freq=1)

    base_learning_rate = 0.001
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=base_learning_rate, momentum=0.9),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])


    initial_epochs = 200

    #loss0, accuracy0 = model.evaluate(validate)

    history_fine = model.fit(train,
                            epochs=initial_epochs,
                            validation_data=validate,
                            callbacks=[model_checkpoint_callback, early])

    base_model.trainable = True
    # Let's take a look to see how many layers are in the base model
    #print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=base_learning_rate,momentum=0.9),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    fine_tune_epochs = 100

    history_fine1 = model.fit(train,
                            epochs=fine_tune_epochs,
                            validation_data=validate,
                            callbacks=[model_checkpoint_callback, early])


    #acc =  history_fine.history['accuracy']
    #acc1= acc + history_fine1.history['accuracy']
    #val_acc = history_fine.history['val_accuracy']
    #val_acc1 = val_acc + history_fine1.history['val_accuracy']

    #loss = history_fine.history['loss']
    #loss1 = loss + history_fine1.history['loss']
    #val_loss = history_fine.history['val_loss']
    #val_loss1 = val_loss + history_fine1.history['val_loss']

    model.load_weights(checkpoint_filepath)
    model.save(model_name)
    return history_fine, history_fine1, early

def plot(history_fine, history_fine1, early):

    acc =  history_fine.history['accuracy']
    acc1= acc + history_fine1.history['accuracy']
    val_acc = history_fine.history['val_accuracy']
    val_acc1 = val_acc + history_fine1.history['val_accuracy']

    loss = history_fine.history['loss']
    loss1 = loss + history_fine1.history['loss']
    val_loss = history_fine.history['val_loss']
    val_loss1 = val_loss + history_fine1.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc1, label='Training Accuracy')
    plt.plot(val_acc1, label='Validation Accuracy')
    plt.ylim([0, 1])
    plt.plot([early.stopped_epoch,early.stopped_epoch],
    plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(loss1, label='Training Loss')
    plt.plot(val_loss1, label='Validation Loss')
    plt.ylim([0, 2.0])
    plt.plot([early.stopped_epoch,early.stopped_epoch],
    plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.grid(True)
    plt.show()