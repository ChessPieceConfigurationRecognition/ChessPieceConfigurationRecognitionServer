import os
import file_manager as fm
from collections import defaultdict
import re
import random
import shutil
from threading import Thread, Event
import sys
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Dropout
import numpy as np
import matplotlib.pyplot as plt
import predict


TRAINING_COMPLETED_MSG = "Training completed."
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_SPLIT_SIZE_TEST = 0.1
DATA_SPLIT_SIZE_TRAIN = 0.7
DATA_SPLIT_SIZE_VALID = 0.2
PRESENCE_EPOCH_COUNT = 5
COLOR_EPOCH_COUNT = 20
TYPE_EPOCH_COUNT = 50
PRESENCE_LEARNING_RATE = 0.001
COLOR_LEARNING_RATE = 0.0009
TYPE_LEARNING_RATE = 0.0008
OUTPUT_PICTURES = ['presence_accuracy.png', 'presence_loss.png', 'presence_confusion_matrix.png', 'color_accuracy.png',
                   'color_loss.png', 'color_confusion_matrix.png', 'type_accuracy.png', 'type_loss.png',
                   'type_confusion_matrix.png']


def strip_ansi_escape_codes(text):
    ansi_escape = re.compile(r'[^a-zA-Z0-9\-:/.%\s]+')
    return ansi_escape.sub('', text)


class OutputCapture:
    def __init__(self):
        self.output = []
        self.current_epoch = ""

    def write(self, text):
        if text.strip():
            self.output.append(text)

    def flush(self):
        pass

    def get_output(self):
        cleaned_output = []
        for line in self.output:
            cleaned_line = strip_ansi_escape_codes(line.replace('\r', '')).strip()
            if cleaned_line.startswith("Epoch"):
                self.current_epoch = cleaned_line
            else:
                match = re.search(r'accuracy:\s*([0-9.]+)\s+-\s+loss:\s*([0-9.]+)', cleaned_line)
                if match:
                    accuracy = match.group(1)
                    loss = match.group(2)
                    cleaned_output.append({
                        'epoch': self.current_epoch,
                        'accuracy': accuracy,
                        'loss': loss
                    })
        return cleaned_output

    def clear_output(self):
        self.output = []


def categorize_files(directory):
    categories = defaultdict(list)
    empty_pattern = re.compile(r'^empty\d+')
    type_color_pattern = re.compile(r'^(pawn|bishop|knight|rook|queen|king)_(white|black)_(white|black)_\d+')
    for filename in os.listdir(directory):
        if empty_pattern.match(filename):
            categories['empty'].append(filename)
        elif type_color_pattern.match(filename):
            match = type_color_pattern.match(filename)
            type_ = match.group(1)
            color1 = match.group(2)
            color2 = match.group(3)
            category = f"{type_}_{color1}_{color2}"
            categories[category].append(filename)
    return categories


def prepare_presence_training(raw_data_path, presence_data_path, files_by_category):
    train_path = fm.create_directory('train', presence_data_path)
    valid_path = fm.create_directory('validation', presence_data_path)
    test_path = fm.create_directory('test', presence_data_path)
    empty_train_path = fm.create_directory('empty', train_path)
    piece_train_path = fm.create_directory('piece', train_path)
    empty_validation_path = fm.create_directory('empty', valid_path)
    piece_validation_path = fm.create_directory('piece', valid_path)
    empty_test_path = fm.create_directory('empty', test_path)
    piece_test_path = fm.create_directory('piece', test_path)

    for category in files_by_category:
        random.shuffle(files_by_category[category])

    for category, files in files_by_category.items():
        total_files = len(files)
        if category == 'empty':
            train_count = int(DATA_SPLIT_SIZE_TRAIN * total_files)
            valid_count = int(DATA_SPLIT_SIZE_VALID * total_files)
            for file in files[:train_count]:
                shutil.copy(os.path.join(raw_data_path, str(file)), empty_train_path)
            for file in files[train_count:train_count + valid_count]:
                shutil.copy(os.path.join(raw_data_path, str(file)), empty_validation_path)
            for file in files[train_count + valid_count:]:
                shutil.copy(os.path.join(raw_data_path, str(file)), empty_test_path)
        else:
            train_count = int(DATA_SPLIT_SIZE_TRAIN * total_files)
            valid_count = int(DATA_SPLIT_SIZE_VALID * total_files)
            for file in files[:train_count]:
                shutil.copy(os.path.join(raw_data_path, str(file)), piece_train_path)
            for file in files[train_count:train_count + valid_count]:
                shutil.copy(os.path.join(raw_data_path, str(file)), piece_validation_path)
            for file in files[train_count + valid_count:]:
                shutil.copy(os.path.join(raw_data_path, str(file)), piece_test_path)


def prepare_color_training(raw_data_path, color_data_path, files_by_category):
    train_path = fm.create_directory('train', color_data_path)
    valid_path = fm.create_directory('validation', color_data_path)
    test_path = fm.create_directory('test', color_data_path)
    black_train_path = fm.create_directory('black', train_path)
    white_train_path = fm.create_directory('white', train_path)
    black_validation_path = fm.create_directory('black', valid_path)
    white_validation_path = fm.create_directory('white', valid_path)
    black_test_path = fm.create_directory('black', test_path)
    white_test_path = fm.create_directory('white', test_path)
    for category in files_by_category:
        random.shuffle(files_by_category[category])

    for category, files in files_by_category.items():
        total_files = len(files)
        if category != 'empty':
            color = category.split('_')[1]
            train_count = int(DATA_SPLIT_SIZE_TRAIN * total_files)
            valid_count = int(DATA_SPLIT_SIZE_VALID * total_files)
            for file in files[:train_count]:
                if color == 'black':
                    shutil.copy(os.path.join(raw_data_path, str(file)), black_train_path)
                elif color == 'white':
                    shutil.copy(os.path.join(raw_data_path, str(file)), white_train_path)
            for file in files[train_count:train_count + valid_count]:
                if color == 'black':
                    shutil.copy(os.path.join(raw_data_path, str(file)), black_validation_path)
                elif color == 'white':
                    shutil.copy(os.path.join(raw_data_path, str(file)), white_validation_path)
            for file in files[train_count + valid_count:]:
                if color == 'black':
                    shutil.copy(os.path.join(raw_data_path, str(file)), black_test_path)
                elif color == 'white':
                    shutil.copy(os.path.join(raw_data_path, str(file)), white_test_path)


def prepare_type_training(raw_data_path, type_data_path, files_by_category):
    train_path = fm.create_directory('train', type_data_path)
    valid_path = fm.create_directory('validation', type_data_path)
    test_path = fm.create_directory('test', type_data_path)

    piece_types = ['pawn', 'rook', 'knight', 'king', 'queen', 'bishop']
    directories = {'train': {}, 'validation': {}, 'test': {}}

    for piece_type in piece_types:
        directories['train'][piece_type] = fm.create_directory(piece_type, train_path)
        directories['validation'][piece_type] = fm.create_directory(piece_type, valid_path)
        directories['test'][piece_type] = fm.create_directory(piece_type, test_path)

    for category in files_by_category:
        random.shuffle(files_by_category[category])

    for category, files in files_by_category.items():
        if category != 'empty':
            piece_type = category.split('_')[0]
            total_files = len(files)
            train_count = int(DATA_SPLIT_SIZE_TRAIN * total_files)
            valid_count = int(DATA_SPLIT_SIZE_VALID * total_files)
            test_count = total_files - train_count - valid_count

            for phase, count in zip(['train', 'validation', 'test'], [train_count, valid_count, test_count]):
                for file in files[:count]:
                    shutil.copy(os.path.join(raw_data_path, str(file)), directories[phase][piece_type])
                files = files[count:]


def generate_graph_data(logs):
    last_entries = []
    last_epoch = -1
    for entry in logs:
        epoch = int(entry['epoch'].split()[1].split('/')[0])
        if last_epoch == -1 or epoch < last_epoch:
            last_entries.append({})
        last_entries[-1][epoch] = entry
        last_epoch = epoch
    while len(last_entries) < 3:
        last_entries.append({})
    return list(last_entries[0].values()), list(last_entries[1].values()), list(last_entries[2].values())


def generate_graph(data, name, path_to_save):
    capitalized_name = name.capitalize()
    epochs = list(range(1, len(data) + 1))
    accuracy = [float(entry['accuracy']) for entry in data]
    loss = [float(entry['loss']) for entry in data]
    accuracy_graph_path = f'{path_to_save}/{name}_accuracy.png'
    loss_graph_path = f'{path_to_save}/{name}_loss.png'
    num_epochs = len(epochs)
    if num_epochs > 10:
        tick_interval = num_epochs // 10
    else:
        tick_interval = 1
    ticks = list(range(1, num_epochs + 1, tick_interval))
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracy, marker='o', linestyle='-', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{capitalized_name} accuracy / epochs')
    plt.xticks(ticks)
    plt.grid(True)
    plt.savefig(accuracy_graph_path)
    plt.close()
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, marker='o', linestyle='-', color='r')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{capitalized_name} loss / epochs')
    plt.xticks(ticks)
    plt.grid(True)
    plt.savefig(loss_graph_path)
    plt.close()


def train_model(presence_dataset_path, color_dataset_path, type_dataset_path, output_path, outputc_path, queue):
    output_capture = OutputCapture()
    thread_event = Event()
    thread_event.set()
    logs = []
    thread = Thread(target=training_update_sender, args=(output_capture, queue, thread_event, logs))
    try:
        sys.stdout = output_capture
        thread.start()
        queue.put("Starting training for the first model (presence of piece) (1/3)")
        train_presence_model(presence_dataset_path, output_path)
        while not queue.empty():
            queue.get()
        time.sleep(5)
        queue.put("Starting training for the second model (color of piece) (2/3)")
        train_color_model(color_dataset_path, output_path)
        while not queue.empty():
            queue.get()
        time.sleep(5)
        queue.put("Starting training for the third model (type of piece) (3/3)")
        train_type_model(type_dataset_path, output_path)
        presence_logs, color_logs, type_logs = generate_graph_data(logs)
        generate_graph(presence_logs, 'presence', output_path)
        generate_graph(color_logs, 'color', output_path)
        generate_graph(type_logs, 'type', output_path)
        outputc_path = fm.move_vgg16_completed_training(output_path, outputc_path)
        predict.generate_confusion_matrixes(presence_dataset_path, color_dataset_path, type_dataset_path, outputc_path)
    finally:
        thread_event.clear()
        thread.join()
        sys.stdout = sys.__stdout__


def training_update_sender(output_capture, queue, event, logs):
    last_message = None
    while event.is_set():
        if len(output_capture.get_output()) > 0 and last_message != output_capture.get_output()[-1]:
            last_message = output_capture.get_output()[-1]
            queue.put(last_message)
            logs.append(last_message)
        time.sleep(2)
    while not queue.empty():
        queue.get()
    queue.put(TRAINING_COMPLETED_MSG)


def train_presence_model(presence_dataset_path, output_path):
    train_dir = os.path.join(presence_dataset_path, 'train')
    validation_dir = os.path.join(presence_dataset_path, 'validation')
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=PRESENCE_LEARNING_RATE), loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(
        train_generator,
        verbose=1,
        epochs=PRESENCE_EPOCH_COUNT,
        validation_data=validation_generator
    )
    model.save(os.path.join(output_path, 'presence_classifier.h5'))


def train_color_model(color_dataset_path, output_path):
    train_dir = os.path.join(color_dataset_path, 'train')
    validation_dir = os.path.join(color_dataset_path, 'validation')
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=COLOR_LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        train_generator,
        verbose=1,
        epochs=COLOR_EPOCH_COUNT,
        validation_data=validation_generator
    )
    model.save(os.path.join(output_path, 'color_classifier.h5'))


def train_type_model(type_dataset_path, output_path):
    train_dir = os.path.join(type_dataset_path, 'train')
    validation_dir = os.path.join(type_dataset_path, 'validation')
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )
    base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(train_generator.num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=x)
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=TYPE_LEARNING_RATE), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(
        train_generator,
        verbose=1,
        epochs=TYPE_EPOCH_COUNT,
        validation_data=validation_generator
    )
    model.save(os.path.join(output_path, 'type_classifier.h5'))
    class_indices = train_generator.class_indices
    indices_path = os.path.join(output_path, 'type_indices.npy')
    np.save(indices_path, class_indices)


def get_training_pictures(training_name, outputc_path):
    pictures = []
    directory = os.path.join(outputc_path, training_name)
    if os.path.exists(directory):
        for filename in OUTPUT_PICTURES:
            picture_path = os.path.join(str(directory), filename)
            if os.path.isfile(picture_path):
                pictures.append(picture_path)
    return pictures
