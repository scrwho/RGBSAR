##!/usr/bin/env python
# coding: utf-8

# # Section A: Model


from ModelsListDiffFuntions import *
import ModelsListDiffFuntions
import nbimporter
import import_ipynb


import os
import sys


def add_path_to_sys(path):
    module_path = os.path.abspath(path)
    if module_path not in sys.path:
        sys.path.append(module_path)


usePath = os.path.join(r'c:', os.sep, 'Users', 'scrwh',
                       'Documents', 'PythonScripts')
add_path_to_sys(usePath)


# List all the functions defined
# print(dir(ModelsListDiffFuntions))


pc_memory_info = get_system_memory()
print("PC RAM memory:", pc_memory_info, "\n")
intel_gpu_memory_info = get_intel_gpu_memory()
print("intel_gpu_memory_info:", intel_gpu_memory_info, "\n")
nvidia_gpu_memory_info = get_nvidia_gpu_memory()
print("nvidia_gpu_memory_info:", nvidia_gpu_memory_info)

tpu_memory_info = get_tpu_memory()
print("TPU memory:", tpu_memory_info)


# ## Models 1


def custom_model(data_dir, batch_size=32, epochs=10, img_size=244, num_classes=9):

    # Build the model architecture
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), input_shape=(
        img_size, img_size, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def evaluate_models(data_dir, batch_size=32, epochs=10, img_size=244):
    # Generate the Folder with the current date and time
    foldername = os.path.join(
        'Models', 'TrainingData1').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    Folder = f'{foldername}_{now}'
    os.makedirs(Folder, exist_ok=True)

    # Check if a GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("GPU available, training on GPU...")
        device_name = tf.test.gpu_device_name()
    else:
        print("GPU not available, training on CPU...")
        device_name = "/CPU:0"

    # Data augmentation and generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,  width_shift_range=0.2,
                                       height_shift_range=0.2,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Data augmentation for validation set
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training, validation and test set
    train_generator = load_data_generator(
        train_datagen, data_dir, 'train', img_size=img_size, batch_size=batch_size)
    val_generator = load_data_generator(
        val_datagen, data_dir, 'val', img_size=img_size, batch_size=batch_size)
    test_generator = load_data_generator(
        val_datagen, data_dir, 'test', img_size=img_size, batch_size=batch_size)

    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

    # Model creation
    models = [('Inception-V2', InceptionV3, 'imagenet'), ('ResNet-50', ResNet50, 'imagenet'),
              ('ResNet-101', ResNet101, 'imagenet'), ('Inception-ResNet-V2',
                                                      InceptionResNetV2, 'imagenet'),
              ('VGG', VGG16, 'imagenet'), ('Custom', custom_model, None)]

    model_metrics = {'Model': [], 'Training Accuracy': [],
                     'Validation Accuracy': [], 'Test Accuracy': []}

    for name, model_fn, weights in models:
        if model_fn == custom_model:
            model = model_fn(data_dir=data_dir, batch_size=batch_size,
                             epochs=epochs, img_size=img_size, num_classes=num_classes)
        else:
            base_model = model_fn(input_shape=(
                img_size, img_size, 3), include_top=False, weights=weights)
            x = Flatten()(base_model.output)
            x = Dense(units=512, activation='relu')(x)
            x = Dense(units=256, activation='relu')(x)
            output = Dense(train_generator.num_classes,
                           activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=output)

            for layer in base_model.layers:
                layer.trainable = False

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])

        # Move the model to the GPU if available
        with tf.device(device_name):
            # Define the learning rate schedule
            def lr_schedule(epoch):
                learning_rate = 0.0001
                if epoch > 30:
                    learning_rate *= 0.1
                elif epoch > 20:
                    learning_rate *= 0.01
                print('Learning rate:', learning_rate)
                print(get_nvidia_gpu_memory())
                return learning_rate

            # Define the callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
            lr_scheduler = LearningRateScheduler(lr_schedule)
            checkpoint = ModelCheckpoint(
                f'{Folder}/best_{name}_model1.h5', monitor='val_loss', save_best_only=True)

            # Train the model
            # print('Training Model:',name)
            print(f'Training Model: {name}')
            start_times = datetime.now()
            print(f'{name} Started: {start_times}')
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=len(val_generator),
                callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
            )
            end_times = datetime.now()
            print(f'{name} Ended: {end_times}')
            print(f'Duration: {end_times - start_times}')

            # Run the garbage collector
            gc.collect()

            now = datetime.now().strftime('%Y-%m-%d %H%M')
            # save your model and its history to disk
            model.save(f'{Folder}/wind_turbine_{name}_model1_{now}.h5')
            with open(f'{Folder}/wind_turbine_{name}_history1_{now}.pkl', 'wb') as f:
                pickle.dump(history.history, f)

        # labels = list(train_generator.class_indices.keys())
        # model.summary()

        # Evaluation
        _, train_acc = model.evaluate(train_generator)
        _, val_acc = model.evaluate(val_generator)
        _, test_acc = model.evaluate(test_generator)
        model_metrics['Model'].append(name)
        model_metrics['Training Accuracy'].append(train_acc)
        model_metrics['Validation Accuracy'].append(val_acc)
        model_metrics['Test Accuracy'].append(test_acc)

    # Saving the performance in a DataFrame
    df = pd.DataFrame(model_metrics)
    df.set_index('Model', inplace=True)
    df['Average'] = df.select_dtypes(include='number').mean(axis=1)

    # Generate the filename with the current date and time
    name = os.path.join(Folder, 'TrainingData1').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    filename = f'{name}_{now}.csv'

    # Save the DataFrame to the CSV file
    df.to_csv(filename, index=True)

    print(df)

    # print(f'Saved DataFrame to {filename}')

    return df


# ### Note
# - data4 = Original sizes
# - data4b = 256
# - data4c = 512
# - data4d = 244
# -


data_folder = os.path.join(
    '..', '..', 'data', 'data4d').replace(os.path.sep, '/')
all_files = get_file_list(data_folder)


# ## Models 1 Training


data_folder = os.path.join('..', '..', 'data', 'data4d').replace(os.path.sep, '/')``````
start_times = datetime.now()

data1 = evaluate_models(data_folder, batch_size=32, epochs=50, img_size=244)

end_times = datetime.now()
print(f'\nFinal Duration: {end_times - start_times}')


# ## Models 2
# - Use distributed training
# - multiple GPUs and want to take advantage of distributed training to potentially speed up the process


def evaluate_models1b(data_dir, batch_size=32, epochs=10, img_size=244):
    # Generate the Folder with the current date and time
    foldername = os.path.join(
        'Models', 'TrainingData1b').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    Folder = f'{foldername}_{now}'
    os.makedirs(Folder, exist_ok=True)

    # Determine device to use
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) > 1:
        strategy = tf.distribute.MirroredStrategy()
    elif len(devices) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    # Data augmentation and generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,  width_shift_range=0.2,
                                       height_shift_range=0.2,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Data augmentation for validation set
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training, validation and test set
    train_generator = load_data_generator(
        train_datagen, data_dir, 'train', img_size=img_size, batch_size=batch_size)
    val_generator = load_data_generator(
        val_datagen, data_dir, 'val', img_size=img_size, batch_size=batch_size)
    test_generator = load_data_generator(
        val_datagen, data_dir, 'test', img_size=img_size, batch_size=batch_size)

    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

    # Model creation
    models = [('Inception-V2', InceptionV3, 'imagenet'), ('ResNet-50', ResNet50, 'imagenet'),
              ('ResNet-101', ResNet101, 'imagenet'), ('Inception-ResNet-V2',
                                                      InceptionResNetV2, 'imagenet'),
              ('VGG16', VGG16, 'imagenet'), ('Custom', custom_model, None)]
    # models = [('Inception-V2', InceptionV3, 'imagenet'),('Inception-ResNet-V2', InceptionResNetV2, 'imagenet'), ('Custom', custom_model, None)]

    model_metrics = {'Model': [], 'Training Accuracy': [],
                     'Validation Accuracy': [], 'Test Accuracy': []}

    for name, model_fn, weights in models:
        with strategy.scope():
            if model_fn == custom_model:
                model = model_fn(data_dir=data_dir, batch_size=batch_size,
                                 epochs=epochs, img_size=img_size, num_classes=num_classes)
            else:
                base_model = model_fn(input_shape=(
                    img_size, img_size, 3), include_top=False, weights=weights)
                x = Flatten()(base_model.output)
                x = Dense(units=512, activation='relu')(x)
                x = Dense(units=256, activation='relu')(x)
                output = Dense(train_generator.num_classes,
                               activation='softmax')(x)
                model = Model(inputs=base_model.input, outputs=output)

                for layer in base_model.layers:
                    layer.trainable = False

            # Compile the model
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy', metrics=['accuracy'])

        # Define the learning rate schedule
        def lr_schedule(epoch):
            learning_rate = 0.0001
            if epoch > 30:
                learning_rate *= 0.1
            elif epoch > 20:
                learning_rate *= 0.01
            print('Learning rate:', learning_rate)
            print(get_nvidia_gpu_memory())
            return learning_rate

        # Define the callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        now = datetime.now().strftime('%Y-%m-%d %H%M')
        checkpoint = ModelCheckpoint(
            f'{Folder}/best_{name}_model1b_{now}.h5', monitor='val_loss', save_best_only=True)

        # Train the model
        # print('Training Model:',name)
        print(f'Training Model: {name}')
        start_times = datetime.now()
        print(f'{name} Started: {start_times}')
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
        )
        end_times = datetime.now()
        print(f'{name} Ended: {end_times}')
        print(f'Duration: {end_times - start_times}')

        # Run the garbage collector
        gc.collect()

        now = datetime.now().strftime('%Y-%m-%d %H%M')
        # save your model and its history to disk
        model.save(f'{Folder}/wind_turbine_{name}_model1b_{now}.h5')
        with open(f'{Folder}/wind_turbine_{name}_history1b_{now}.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        # Evaluation
        _, train_acc = model.evaluate(train_generator)
        _, val_acc = model.evaluate(val_generator)
        _, test_acc = model.evaluate(test_generator)
        model_metrics['Model'].append(name)
        model_metrics['Training Accuracy'].append(train_acc)
        model_metrics['Validation Accuracy'].append(val_acc)
        model_metrics['Test Accuracy'].append(test_acc)

    # Saving the performance in a DataFrame
    df = pd.DataFrame(model_metrics)
    df.set_index('Model', inplace=True)
    df['Average'] = df.select_dtypes(include='number').mean(axis=1)

    # Generate the filename with the current date and time
    name = os.path.join(Folder, 'TrainingData1b').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    filename = f'{name}_{now}.csv'

    # Save the DataFrame to the CSV file
    df.to_csv(filename, index=True)

    print(df)

    # print(f'Saved DataFrame to {filename}')
    return df


# ### Models 2 Training


start_times = datetime.now()
data1b = evaluate_models1b(data_folder, batch_size=32, epochs=50, img_size=244)

end_times = datetime.now()
print(f'\nFinal Duration: {end_times - start_times}')


# ## Models 3


def custom_modelB(data_dir, batch_size=32, epochs=10, img_size=244, num_classes=9):

    # Build the model architecture
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(32, (3, 3), input_shape=(
        img_size, img_size, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and dense layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',
              kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def evaluate_modelsb(data_dir, batch_size=32, epochs=10, img_size=244):
    # Generate the Folder with the current date and time
    foldername = os.path.join(
        'Models', 'TrainingData2').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    Folder = f'{foldername}_{now}'
    os.makedirs(Folder, exist_ok=True)

    # Check if a GPU is available
    if tf.config.list_physical_devices('GPU'):
        print("GPU available, training on GPU...")
        device_name = tf.test.gpu_device_name()
    else:
        print("GPU not available, training on CPU...")
        device_name = "/CPU:0"

    # Data augmentation and generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,  width_shift_range=0.2,
                                       height_shift_range=0.2,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Data augmentation for validation set
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training, validation and test set
    train_generator = load_data_generator(
        train_datagen, data_dir, 'train', img_size=img_size, batch_size=batch_size)
    val_generator = load_data_generator(
        val_datagen, data_dir, 'val', img_size=img_size, batch_size=batch_size)
    test_generator = load_data_generator(
        val_datagen, data_dir, 'test', img_size=img_size, batch_size=batch_size)

    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

    # Model creation
    # models = [('Inception-V2', InceptionV3, 'imagenet'), ('ResNet-50', ResNet50, 'imagenet'),
    #           ('ResNet-101', ResNet101, 'imagenet'), ('Inception-ResNet-V2', InceptionResNetV2, 'imagenet'),
    #           ('VGG', VGG16, 'imagenet'), ('Custom', custom_model, None)]
    # models = [('Inception-V2', InceptionV3, 'imagenet'), ('Custom', custom_model, None)]

    models = [('Inception-V2', InceptionV3, 'imagenet'), ('Inception-ResNet-V2',
                                                          InceptionResNetV2, 'imagenet'), ('Custom', custom_model, None)]

    model_metrics = {'Model': [], 'Training Accuracy': [],
                     'Validation Accuracy': [], 'Test Accuracy': []}

    for name, model_fn, weights in models:
        if model_fn == custom_model:
            model = model_fn(data_dir=data_dir, batch_size=batch_size,
                             epochs=epochs, img_size=img_size, num_classes=num_classes)
        else:
            base_model = model_fn(input_shape=(
                img_size, img_size, 3), include_top=False, weights=weights)
            x = Flatten()(base_model.output)
            x = Dense(units=1024, activation='relu')(x)
            x = Dense(units=512, activation='relu')(x)
            output = Dense(train_generator.num_classes,
                           activation='softmax')(x)
            model = Model(inputs=base_model.input, outputs=output)

            for layer in base_model.layers:
                layer.trainable = False

        # Compile the model
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='categorical_crossentropy', metrics=['accuracy'])
        # optimizer = Adam(learning_rate=0.001)
        # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Move the model to the GPU if available
        with tf.device(device_name):
            # Define the learning rate schedule
            def lr_schedule(epoch):
                learning_rate = 0.0001
                if epoch > 30:
                    learning_rate *= 0.1
                elif epoch > 20:
                    learning_rate *= 0.01
                print('Learning rate:', learning_rate)
                print(get_nvidia_gpu_memory())
                return learning_rate

            # Define the callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=5)
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
            lr_scheduler = LearningRateScheduler(lr_schedule)
            checkpoint = ModelCheckpoint(
                f'{Folder}/best_{name}_model2.h5', monitor='val_loss', save_best_only=True)

            # Train the model
            # print('Training Model:',name)
            print(f'Training Model: {name}')
            start_times = datetime.now()
            print(f'{name} Started: {start_times}')
            history = model.fit(
                train_generator,
                steps_per_epoch=len(train_generator),
                epochs=epochs,
                validation_data=val_generator,
                validation_steps=len(val_generator),
                callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
            )
            end_times = datetime.now()
            print(f'{name} Ended: {end_times}')
            print(f'Duration: {end_times - start_times}')

            # Run the garbage collector
            gc.collect()

            now = datetime.now().strftime('%Y-%m-%d %H%M')
            # save your model and its history to disk
            model.save(f'{Folder}/wind_turbine_{name}_model2_{now}.h5')
            with open(f'{Folder}/wind_turbine_{name}_history2_{now}.pkl', 'wb') as f:
                pickle.dump(history.history, f)

        # labels = list(train_generator.class_indices.keys())
        # model.summary()

        # Evaluation
        _, train_acc = model.evaluate(train_generator)
        _, val_acc = model.evaluate(val_generator)
        _, test_acc = model.evaluate(test_generator)
        model_metrics['Model'].append(name)
        model_metrics['Training Accuracy'].append(train_acc)
        model_metrics['Validation Accuracy'].append(val_acc)
        model_metrics['Test Accuracy'].append(test_acc)

    # Saving the performance in a DataFrame
    df = pd.DataFrame(model_metrics)
    df.set_index('Model', inplace=True)
    df['Average'] = df.select_dtypes(include='number').mean(axis=1)

    # Generate the filename with the current date and time
    name = os.path.join(Folder, 'TrainingData2').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    filename = f'{name}_{now}.csv'

    # Save the DataFrame to the CSV file
    df.to_csv(filename, index=True)

    print(df)

    # print(f'Saved DataFrame to {filename}')

    return df


# ## Models 3 Training


start_times = datetime.now()

data2 = evaluate_modelsb(data_folder, batch_size=32, epochs=50, img_size=244)

end_times = datetime.now()
print(f'\nFinal Duration: {end_times - start_times}')


data2


# Run the garbage collector
gc.collect()


# ## Models 4


def evaluate_modelsb2(data_dir, batch_size=32, epochs=10, img_size=244):
    # Generate the Folder with the current date and time
    foldername = os.path.join(
        'Models', 'TrainingData2b').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    Folder = f'{foldername}_{now}'
    os.makedirs(Folder, exist_ok=True)

    # Determine device to use
    devices = tf.config.list_physical_devices('GPU')
    if len(devices) > 1:
        strategy = tf.distribute.MirroredStrategy()
    elif len(devices) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

    # Data augmentation and generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,  width_shift_range=0.2,
                                       height_shift_range=0.2,  shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

    # Data augmentation for validation set
    val_datagen = ImageDataGenerator(rescale=1./255)

    # Load the training, validation and test set
    train_generator = load_data_generator(
        train_datagen, data_dir, 'train', img_size=img_size, batch_size=batch_size)
    val_generator = load_data_generator(
        val_datagen, data_dir, 'val', img_size=img_size, batch_size=batch_size)
    test_generator = load_data_generator(
        val_datagen, data_dir, 'test', img_size=img_size, batch_size=batch_size)

    num_classes = len(train_generator.class_indices)
    labels = list(train_generator.class_indices.keys())

    # Model creation
    # models = [('Inception-V2', InceptionV3, 'imagenet'), ('ResNet-50', ResNet50, 'imagenet'),
    #           ('ResNet-101', ResNet101, 'imagenet'), ('Inception-ResNet-V2', InceptionResNetV2, 'imagenet'),
    #           ('VGG16', VGG16, 'imagenet'), ('Custom', custom_modelB, None)]
    models = [('Inception-V2', InceptionV3, 'imagenet'), ('Inception-ResNet-V2',
                                                          InceptionResNetV2, 'imagenet'), ('Custom', custom_model, None)]

    model_metrics = {'Model': [], 'Training Accuracy': [],
                     'Validation Accuracy': [], 'Test Accuracy': []}

    for name, model_fn, weights in models:
        with strategy.scope():
            if model_fn == custom_modelB:
                model = model_fn(data_dir=data_dir, batch_size=batch_size,
                                 epochs=epochs, img_size=img_size, num_classes=num_classes)
            else:
                base_model = model_fn(input_shape=(
                    img_size, img_size, 3), include_top=False, weights=weights)
                x = Flatten()(base_model.output)
                x = Dense(units=1024, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(x)
                x = Dropout(0.5)(x)
                x = Dense(units=512, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001))(x)
                x = Dropout(0.5)(x)
                output = Dense(units=train_generator.num_classes,
                               activation='softmax')(x)
                model = Model(inputs=base_model.input, outputs=output)

                for layer in base_model.layers:
                    layer.trainable = False

            # Compile the model
            model.compile(optimizer=Adam(learning_rate=0.001),
                          loss='categorical_crossentropy', metrics=['accuracy'])

        # Define the learning rate schedule

        def lr_schedule(epoch):
            learning_rate = 0.0001
            if epoch > 30:
                learning_rate *= 0.1
            elif epoch > 20:
                learning_rate *= 0.01
            print('Learning rate:', learning_rate)
            print(get_nvidia_gpu_memory())
            return learning_rate

        # Define the callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        now = datetime.now().strftime('%Y-%m-%d %H%M')
        checkpoint = ModelCheckpoint(
            f'{Folder}/best_{name}_model2b_{now}.h5', monitor='val_loss', save_best_only=True)

        # Train the model
        # print('Training Model:',name)
        print(f'Training Model: {name}')
        start_times = datetime.now()
        print(f'{name} Started: {start_times}')
        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=[early_stopping, reduce_lr, lr_scheduler, checkpoint]
        )
        end_times = datetime.now()
        print(f'{name} Ended: {end_times}')
        print(f'Duration: {end_times - start_times}')

        # Run the garbage collector
        gc.collect()

        now = datetime.now().strftime('%Y-%m-%d %H%M')
        # save your model and its history to disk
        model.save(f'{Folder}/wind_turbine_{name}_model2b_{now}.h5')
        with open(f'{Folder}/wind_turbine_{name}_history2b_{now}.pkl', 'wb') as f:
            pickle.dump(history.history, f)

        # labels = list(train_generator.class_indices.keys())
        # model.summary()

        # Evaluation
        _, train_acc = model.evaluate(train_generator)
        _, val_acc = model.evaluate(val_generator)
        _, test_acc = model.evaluate(test_generator)
        model_metrics['Model'].append(name)
        model_metrics['Training Accuracy'].append(train_acc)
        model_metrics['Validation Accuracy'].append(val_acc)
        model_metrics['Test Accuracy'].append(test_acc)

    # Saving the performance in a DataFrame
    df = pd.DataFrame(model_metrics)
    df.set_index('Model', inplace=True)
    df['Average'] = df.select_dtypes(include='number').mean(axis=1)

    # Generate the filename with the current date and time
    name = os.path.join(Folder, 'TrainingData2b').replace(os.path.sep, '/')
    now = datetime.now().strftime('%Y-%m-%d %H%M')
    filename = f'{name}_{now}.csv'

    # Save the DataFrame to the CSV file
    df.to_csv(filename, index=True)

    print(df)

    # print(f'Saved DataFrame to {filename}')
    return df


# ## Models 3 Training


start_times = datetime.now()

data2b = evaluate_modelsb2(data_folder, batch_size=32, epochs=50, img_size=244)

end_times = datetime.now()
print(f'\nFinal Duration: {end_times - start_times}')


# Run the garbage collector
gc.collect()
