import os
import cv2
import shutil
from tqdm import tqdm

from config import (
    raw_dataset, 
    resized_dataset, 
    image_height, 
    image_width,
    target_datasets
    )


def resize_dataset(target_folder: str, image_height: int, image_width: int) -> None:
    filenames = os.listdir(target_folder)
    cutted_filenames = [filename.split('.')[0] for filename in filenames]
    unique_objects = list(set(cutted_filenames))

    for unique_object in tqdm(unique_objects):
        try:
            image_path = os.path.join(target_folder, unique_object + '.jpg')
            image = cv2.imread(image_path)
            height, width = image.shape[0], image.shape[1]
            if (height != image_height) or (width != image_width):
                image = cv2.resize(image, (image_width, image_height), interpolation = cv2.INTER_LINEAR)
                cv2.imwrite(image_path, image) 
            else:
                continue
            
        except FileNotFoundError as e:
            print(e)
            continue
        
        except Exception as e:
            print(e)
            continue


def main():
    if os.path.isdir(resized_dataset):
        shutil.rmtree(resized_dataset)
    
    print(f"Копирование {raw_dataset} в {resized_dataset}")
    shutil.copytree(raw_dataset, resized_dataset)
    print("Копирование окончено")
    
    for target_dataset in target_datasets:
        print(f"Начало предобработки {target_dataset}")
        resize_dataset(target_folder=target_dataset, image_height=image_height, image_width=image_width)
        print(f"Обработка {target_dataset} закончена")

if __name__ == "__main__":
    main()