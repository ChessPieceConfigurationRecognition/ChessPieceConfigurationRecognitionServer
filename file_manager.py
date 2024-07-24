import os
import shutil


def get_current_dir_count(directory):
    try:
        entries = os.listdir(directory)
        directories = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        return len(directories)
    except Exception as e:
        print(f"Exception: {e}")
        return 0


def save_files(files, new_dir_name, upload_dir):
    for file in files:
        split_path = file.filename.split('/')
        file_path = os.path.join(upload_dir, new_dir_name, '/'.join(split_path[1:]))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
    return os.path.join(upload_dir, new_dir_name)


def create_directory(new_dir_name, output_dir):
    new_dir_path = os.path.join(output_dir, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)
    return new_dir_path


def get_absolute_path(relative_path):
    current_working_directory = os.getcwd()
    absolute_path = os.path.abspath(os.path.join(current_working_directory, relative_path))
    return absolute_path


def is_name_used(name, output_dir):
    folder_path = os.path.join(output_dir, name)
    return os.path.isdir(folder_path)


def move_yolo_completed_training(training_name, output_dir, outputc_dir):
    source_path = os.path.join(output_dir, training_name)
    destination_path = os.path.join(outputc_dir, training_name)
    shutil.move(str(source_path), str(destination_path))


def move_vgg16_completed_training(output_dir, outputc_dir):
    shutil.move(output_dir, outputc_dir)
    moved_folder_path = os.path.join(outputc_dir, os.path.basename(output_dir))
    return moved_folder_path


def get_completed_training_folders(outputc_dir):
    items = os.listdir(outputc_dir)
    folders = [item for item in items if os.path.isdir(os.path.join(outputc_dir, item))]
    return folders
