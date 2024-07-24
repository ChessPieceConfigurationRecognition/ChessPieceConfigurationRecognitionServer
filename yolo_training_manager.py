from ultralytics import YOLO
from threading import Thread, Event
import file_manager as fm
import os
import time
import re
import sys


TRAINING_COMPLETED_MSG = "Training completed."
EPOCH_COUNT = 10
IMAGE_SIZE = 1024
OUTPUT_PICTURES = ['labels.jpg', 'confusion_matrix.png', 'confusion_matrix_normalized.png', 'results.png']


def strip_ansi_escape_codes(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


class OutputCapture:
    def __init__(self):
        self.output = []

    def write(self, text):
        if text.strip():
            self.output.append(text)

    def flush(self):
        pass

    def get_output(self):
        cleaned_output = [strip_ansi_escape_codes(line.replace('\r', '')) for line in self.output]
        pattern = re.compile(r'^\s*\d/\d\s+')
        matching_lines = [line for line in cleaned_output if pattern.match(line)]
        formatted_lines = []

        for line in matching_lines:
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 8:
                progress_match = re.match(r'^\d{1,3}', parts[7])
                progress_value = progress_match.group(0) if progress_match else parts[7]
                formatted_line = {
                    "epoch": parts[0],
                    "gpu_mem": parts[1],
                    "box_loss": parts[2],
                    "cls_loss": parts[3],
                    "dfl_loss": parts[4],
                    "instances": parts[5],
                    "progress": progress_value
                }
                if len(parts) > 8:
                    speed = " ".join(parts[8:][2:])
                    formatted_line["speed"] = speed.strip("[]")
                formatted_lines.append(formatted_line)
        return formatted_lines


def train_model(dataset_path, output_path, outputc_path, training_name, queue):
    output_capture = OutputCapture()
    thread_event = Event()
    thread_event.set()
    thread = Thread(target=training_update_sender, args=(output_capture, queue, thread_event))
    try:
        model = YOLO("yolov8s.pt")
        abs_dataset_path = fm.get_absolute_path(os.path.join(dataset_path, 'data.yaml'))
        abs_output_path = fm.get_absolute_path(output_path)
        sys.stdout = output_capture
        sys.stderr = output_capture
        thread.start()
        queue.put("Started training.")
        model.train(data=abs_dataset_path, epochs=EPOCH_COUNT, imgsz=IMAGE_SIZE, name=training_name,
                    project=abs_output_path)
        fm.move_yolo_completed_training(training_name, output_path, outputc_path)
    finally:
        thread_event.clear()
        thread.join()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def training_update_sender(output_capture, queue, event):
    last_message = None
    while event.is_set():
        if len(output_capture.get_output()) > 0 and last_message != output_capture.get_output()[-1]:
            last_message = output_capture.get_output()[-1]
            queue.put(last_message)
        time.sleep(2)
    queue.put(TRAINING_COMPLETED_MSG)


def get_training_pictures(training_name, outputc_path):
    pictures = []
    directory = os.path.join(outputc_path, training_name)

    if os.path.exists(directory):
        for filename in OUTPUT_PICTURES:
            picture_path = os.path.join(str(directory), filename)
            if os.path.isfile(picture_path):
                pictures.append(picture_path)

    return pictures
