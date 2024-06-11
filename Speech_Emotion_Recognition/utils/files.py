"""organize the dataset"""

import os, shutil

def remove(file_path: str) -> None:
    """delete all the file which is not `.wav`"""
    for root, dirs, files in os.walk(file_path):
        for item in files:
            if not item.endswith('.wav'):
                try:
                    print("Delete file: ", os.path.join(root, item))
                    os.remove(os.path.join(root, item))
                except:
                    continue

def rename(file_path: str) -> None:
    """Change the name to specify format"""
    for root, dirs, files in os.walk(file_path):
        for item in files:
            if item.endswith('.wav'):
                people_name = root.split('/')[-2]
                emotion_name = root.split('/')[-1]
                item_name = item[:-4] # audio name delete ".wav"
                old_path = os.path.join(root, item)
                new_path = os.path.join(root, item_name + '-' + emotion_name + '-'+ people_name + '.wav') # new audio path
                try:
                    os.rename(old_path, new_path)
                    print('converting ', old_path, ' to ', new_path)
                except:
                    continue

def move(file_path: str) -> None:
    """Categorize the audio by emotion and place it in different folders"""
    for root, dirs, files in os.walk(file_path):
        for item in files:
            if item.endswith('.wav'):
                emotion_name = root.split('/')[-1]
                old_path = os.path.join(root, item)
                new_path = os.path.join(file_path, emotion_name, item)
                try:
                    shutil.move(old_path, new_path)
                    print("Move ", old_path, " to ", new_path)
                except:
                    continue

def mkdirs(folder_path: str) -> None:
    """check if the folder exist, if not creat a new one"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
