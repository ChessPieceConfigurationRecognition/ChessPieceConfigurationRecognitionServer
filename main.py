import multiprocessing
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
import yolo_training_manager as ytm
import vgg16_training_manager as vtm
import file_manager as fm
from flask import Request
import os
import base64
import json
import numpy as np
import cv2
import predict as pred


class CustomRequest(Request):
    def __init__(self, *args, **kwargs):
        super(CustomRequest, self).__init__(*args, **kwargs)
        self.max_form_parts = 30000


app = Flask(__name__)
app.request_class = CustomRequest
app.config['MAX_CONTENT_LENGTH'] = 128 * 1024 * 1024
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

UPLOAD_DIR = 'uploads'
OUTPUT_DIR = 'outputs'
YOLO_OUTPUT_COMPLETED_DIR = 'outputs_completed'
VGG16_MODELS_DIR = 'models'
VGG16_MODELS_COMPLETED_DIR = 'models_completed'
VGG16_PRESENCE_MODEL_DIR = 'presence'
VGG16_COLOR_MODEL_DIR = 'color'
VGG16_TYPE_MODEL_DIR = 'type'

training_process = multiprocessing.Process()
training_logs = []
training_running = False


@app.route('/train', methods=['POST'])
def train():
    global training_process, training_running, training_logs
    if 'files' not in request.files:
        return jsonify({'error': 'No files part'}), 400

    files = request.files.getlist('files')
    training_name = request.form.get('name')
    training_type = request.form.get('type')

    if (fm.is_name_used(training_name, YOLO_OUTPUT_COMPLETED_DIR) or
            fm.is_name_used('model_' + training_name, VGG16_MODELS_COMPLETED_DIR)
            or fm.is_name_used('dataset_' + training_name, UPLOAD_DIR)):
        return jsonify({'error': 'Please choose another name for your model.'}), 400

    if training_type == 'object_detection':
        new_dir_name = "dataset_" + training_name
        output_dir_path = fm.create_directory("output_" + training_name, OUTPUT_DIR)
        dataset_dir_path = fm.save_files(files, new_dir_name, UPLOAD_DIR)

        if training_process is None or not training_process.is_alive():
            training_queue = multiprocessing.Queue()
            training_running = True
            socketio.start_background_task(training_update_receiver, training_queue, 'yolo')
            training_process = multiprocessing.Process(target=ytm.train_model, args=(dataset_dir_path,
                                                                                     output_dir_path,
                                                                                     YOLO_OUTPUT_COMPLETED_DIR,
                                                                                     training_name,
                                                                                     training_queue))

            training_process.start()
            return jsonify({'message': 'Training started'}), 200
        else:
            return jsonify({'message': 'Training is already running'}), 400
    elif training_type == 'image_classification':
        new_dir_name = "dataset_" + training_name
        outer_dataset_dir_path = fm.create_directory(new_dir_name, UPLOAD_DIR)
        dataset_dir_path = fm.save_files(files, 'raw', outer_dataset_dir_path)
        presence_dataset_dir_path = fm.create_directory(VGG16_PRESENCE_MODEL_DIR, outer_dataset_dir_path)
        color_dataset_dir_path = fm.create_directory(VGG16_COLOR_MODEL_DIR, outer_dataset_dir_path)
        type_dataset_dir_path = fm.create_directory(VGG16_TYPE_MODEL_DIR, outer_dataset_dir_path)
        files_by_category = vtm.categorize_files(dataset_dir_path)
        model_path = fm.create_directory('model_' + training_name, VGG16_MODELS_DIR)
        vtm.prepare_presence_training(dataset_dir_path, presence_dataset_dir_path, files_by_category)
        vtm.prepare_color_training(dataset_dir_path, color_dataset_dir_path, files_by_category)
        vtm.prepare_type_training(dataset_dir_path, type_dataset_dir_path, files_by_category)
        if training_process is None or not training_process.is_alive():
            training_queue = multiprocessing.Queue()
            training_running = True
            socketio.start_background_task(training_update_receiver, training_queue, 'vgg16')

            training_process = multiprocessing.Process(target=vtm.train_model, args=(presence_dataset_dir_path,
                                                                                     color_dataset_dir_path,
                                                                                     type_dataset_dir_path,
                                                                                     model_path,
                                                                                     VGG16_MODELS_COMPLETED_DIR,
                                                                                     training_queue))
            training_process.start()

            return jsonify({'message': 'Training started'}), 200
        else:
            return jsonify({'message': 'Training is already running'}), 400


def training_update_receiver(training_queue, train_type):
    global training_logs, training_running
    while training_running:
        if not training_queue.empty():
            message = training_queue.get()
            training_logs.append(message)
            if message == ytm.TRAINING_COMPLETED_MSG or message == vtm.TRAINING_COMPLETED_MSG:
                socketio.emit('training_update', {'output': message, 'done': True, 'type': train_type})
                break
            else:
                socketio.emit('training_update', {'output': message, 'done': False, 'type': train_type})
        socketio.sleep(2)
    training_logs = []
    training_running = False


@app.route('/stop', methods=['POST'])
def stop_training():
    global training_process, training_running, training_logs
    if training_process is not None and training_process.is_alive():
        training_process.terminate()
        training_process.join()

        training_running = False
        training_logs = []
        return jsonify({'message': 'Training stopped'}), 200
    else:
        return jsonify({'message': 'No training process is running'}), 400


@app.route('/status', methods=['GET'])
def training_status():
    global training_running, training_logs
    return jsonify({'training_running': training_running, 'training_logs': training_logs}), 200


@app.route('/models', methods=['GET'])
def existing_models():
    vgg16_models = fm.get_completed_training_folders(VGG16_MODELS_COMPLETED_DIR)
    yolo_models = fm.get_completed_training_folders(YOLO_OUTPUT_COMPLETED_DIR)
    combined_models = []
    for model in vgg16_models:
        name = model.replace("model_", "", 1)
        combined_models.append({'name': name, 'type': 'vgg16'})
    for model in yolo_models:
        combined_models.append({'name': model, 'type': 'yolo'})

    return jsonify({'model_list': combined_models}), 200


@app.route('/model_data', methods=['GET'])
def get_model_data():
    training_name = request.args.get('name')
    training_type = request.args.get('type')
    pictures = []
    if training_type == 'yolo':
        pictures = ytm.get_training_pictures(training_name, YOLO_OUTPUT_COMPLETED_DIR)
    elif training_type == 'vgg16':
        pictures = vtm.get_training_pictures('model_' + training_name, VGG16_MODELS_COMPLETED_DIR)
    if pictures:
        images_data = {}
        for image_path in pictures:
            with open(image_path, 'rb') as file:
                image_name = os.path.basename(image_path)
                base64_string = base64.b64encode(file.read()).decode('utf-8')
                images_data[image_name] = base64_string
        ordered_images_data = [images_data[os.path.basename(image_path)] for image_path in pictures]
        return jsonify({"images": ordered_images_data}), 200
    else:
        return jsonify({'message': 'No images'}), 400


@app.route('/predict', methods=['POST'])
def predict():
    image = request.files['image']
    extra_data = request.form.get('extraData')
    if extra_data:
        extra_data = json.loads(extra_data)
        corners = extra_data.get('corners')
        model_name = extra_data.get('name')
        model_type = extra_data.get('type')
        player = extra_data.get('player')
        nparr = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if model_type == 'vgg16':
            fen = pred.predict_vgg16_chessboard(img, VGG16_MODELS_COMPLETED_DIR, model_name, player, corners)
            return jsonify({'message': fen}), 200
        else:
            fen = pred.predict_yolo_chessboard(img, YOLO_OUTPUT_COMPLETED_DIR, model_name, player, corners)
            return jsonify({'message': fen}), 200
    return jsonify({'message': 'Fail'}), 400


if __name__ == '__main__':
    socketio.run(app, allow_unsafe_werkzeug=True, debug=True, log_output=False, use_reloader=False)
