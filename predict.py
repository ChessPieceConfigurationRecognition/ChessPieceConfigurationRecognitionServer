import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from ultralytics import YOLO


def preprocess_image(image):
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array


def is_piece_present(model, image, threshold=0.1):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    return predictions[0][0] > threshold


def is_piece_white(model, image, threshold=0.25):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    return predictions[0][0] > threshold


def piece_type(model, image, class_indices):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_class = list(class_indices.keys())[class_index]
    return predicted_class


def piece_presence_confusion_matrix(model, presence_dir):
    true_labels = []
    predicted_labels = []
    for sub_dir in ['test']:
        sub_dir_path = os.path.join(presence_dir, sub_dir)
        for label in ['empty', 'piece']:
            label_dir = os.path.join(sub_dir_path, label)
            for filename in os.listdir(label_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(label_dir, filename)
                    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                    prediction = is_piece_present(model, image)
                    true_labels.append(label)
                    predicted_labels.append('piece' if prediction else 'empty')
    confusion_mat = confusion_matrix(true_labels, predicted_labels, labels=['empty', 'piece'])
    return confusion_mat


def piece_color_confusion_matrix(model, presence_dir):
    true_labels = []
    predicted_labels = []
    for sub_dir in ['test']:
        sub_dir_path = os.path.join(presence_dir, sub_dir)
        for label in ['black', 'white']:
            label_dir = os.path.join(sub_dir_path, label)
            for filename in os.listdir(label_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(label_dir, filename)
                    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                    prediction = is_piece_white(model, image)
                    true_labels.append(label)
                    predicted_labels.append('white' if prediction else 'black')

    confusion_mat = confusion_matrix(true_labels, predicted_labels, labels=['black', 'white'])
    return confusion_mat


def piece_type_confusion_matrix(model, type_dataset_path, class_indices):
    true_labels = []
    predicted_labels = []
    for sub_dir in ['test']:
        sub_dir_path = os.path.join(type_dataset_path, sub_dir)
        for class_name in list(class_indices.keys()):
            class_dir = os.path.join(sub_dir_path, class_name)
            for filename in os.listdir(class_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(class_dir, filename)
                    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
                    prediction = piece_type(model, image, class_indices)
                    true_labels.append(class_name)
                    predicted_labels.append(prediction)
    confusion_mat = confusion_matrix(true_labels, predicted_labels, labels=list(class_indices.keys()))
    return confusion_mat


def generate_confusion_matrixes(presence_dataset_path, color_dataset_path, type_dataset_path, outputc_path):
    presence_model = tf.keras.models.load_model(os.path.join(outputc_path, 'presence_classifier.h5'))
    cf_presence = piece_presence_confusion_matrix(presence_model, presence_dataset_path)
    save_presence_confusion_matrix_as_image(cf_presence, os.path.join(outputc_path, 'presence_confusion_matrix.png'))

    color_model = tf.keras.models.load_model(os.path.join(outputc_path, 'color_classifier.h5'))
    cf_color = piece_color_confusion_matrix(color_model, color_dataset_path)
    save_color_confusion_matrix_as_image(cf_color, os.path.join(outputc_path, 'color_confusion_matrix.png'))

    type_model = tf.keras.models.load_model(os.path.join(outputc_path, 'type_classifier.h5'))
    type_indices_path = os.path.join(outputc_path, 'type_indices.npy')
    type_class_indices = np.load(type_indices_path, allow_pickle=True).item()
    cf_type = piece_type_confusion_matrix(type_model, type_dataset_path, type_class_indices)
    save_type_confusion_matrix_as_image(cf_type, os.path.join(outputc_path, 'type_confusion_matrix.png'),
                                        type_class_indices)


def save_presence_confusion_matrix_as_image(cm, output_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title('Presence confusion matrix')
    plt.colorbar()
    plt.xticks(np.arange(2), ['Empty', 'Piece'])
    plt.yticks(np.arange(2), ['Empty', 'Piece'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_color_confusion_matrix_as_image(cm, output_path):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title('Color confusion matrix')
    plt.colorbar()
    plt.xticks(np.arange(2), ['Black', 'White'])
    plt.yticks(np.arange(2), ['Black', 'White'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_type_confusion_matrix_as_image(cm, output_path, class_indices):
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap='Blues')
    plt.title('Type confusion matrix')
    plt.colorbar()
    class_labels = list(class_indices.keys())
    plt.xticks(np.arange(len(class_indices)), class_labels)
    plt.yticks(np.arange(len(class_indices)), class_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    for i in range(len(class_indices)):
        for j in range(len(class_indices)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def preprocess_board_image(image, player, corners):
    if corners is not None:
        src_points = np.float32([[corner['xPercent'] * image.shape[1] / 100, corner['yPercent'] * image.shape[0] / 100]
                                 for corner in corners])
        width, height = 1024, 1024
        dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.warpPerspective(image, perspective_matrix, (width, height))
    if player == 'black':
        image = cv2.flip(image, -1)
    return image


def predict_vgg16_chessboard(image, outputc_path, model_name, corners, player):
    image = preprocess_board_image(image, corners, player)
    square_size = image.shape[0] // 8
    presence_model = tf.keras.models.load_model(os.path.join(outputc_path, f'model_{model_name}',
                                                             'presence_classifier.h5'))
    color_model = tf.keras.models.load_model(os.path.join(outputc_path, f'model_{model_name}', 'color_classifier.h5'))
    type_model = tf.keras.models.load_model(os.path.join(outputc_path, f'model_{model_name}', 'type_classifier.h5'))
    type_indices_path = os.path.join(outputc_path, f'model_{model_name}', 'type_indices.npy')
    type_class_indices = np.load(type_indices_path, allow_pickle=True).item()
    board = []
    for i in range(8):
        board.append([])
        for j in range(8):
            square = image[i * square_size: (i + 1) * square_size, j * square_size: (j + 1) * square_size]
            square_rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(square_rgb)
            resized_image = pil_image.resize((224, 224))
            if is_piece_present(presence_model, resized_image):
                color = 'black'
                if is_piece_white(color_model, resized_image):
                    color = 'white'
                p_type = piece_type(type_model, resized_image, type_class_indices)
                board[-1].append(f'{color}-{p_type}')
            else:
                board[-1].append('empty')
    return pieces_to_fen(board)


def predict_yolo_chessboard(image, outputc_path, model_name, corners, player):
    image = preprocess_board_image(image, corners, player)
    model = YOLO(str(os.path.join(outputc_path, model_name, 'weights', 'best.pt')))
    results = model(image)
    boxes = results[0].boxes.xyxy
    names = results[0].names
    square_width = image.shape[1] // 8
    square_height = image.shape[0] // 8
    board = [['empty' for _ in range(8)] for _ in range(8)]
    box_cnt = 0
    for box in boxes:
        center_x = (box[0] + box[2]) / 2
        center_y = (box[1] + box[3]) / 2
        square_x = int(center_x // square_width)
        square_y = int(center_y // square_height)
        board[square_y][square_x] = names[int(results[0].boxes.cls[box_cnt])]
        box_cnt += 1

    return pieces_to_fen(board)


def pieces_to_fen(pieces):
    line_count = 0
    fen = ''
    for line in pieces:
        empty = 0
        for piece in line:
            if piece == 'empty':
                empty += 1
                continue
            color = piece.find("white")
            split = piece.split('-')
            piece_symbol = split[1][0]
            if split[1][0] == 'k':
                if split[1][1] == 'n':
                    piece_symbol = 'n'
            if color != -1:
                piece_symbol = piece_symbol.upper()
            if empty > 0:
                fen += str(empty)
                empty = 0
            fen += piece_symbol
        if empty > 0:
            fen += str(empty)
        line_count += 1
        if line_count != 8:
            fen += '/'
    return fen
