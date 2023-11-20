import os
import shutil

import cv2
from tqdm import tqdm

from config import image_height, image_width, raw_train_dataset, train_dataset


def resize_dataset(target_folder: str, image_height: int, image_width: int) -> None:
    filenames = os.listdir(target_folder)
    cutted_filenames = [filename.split(".")[0] for filename in filenames]
    unique_objects = list(set(cutted_filenames))

    for unique_object in tqdm(unique_objects):
        try:
            image_path = os.path.join(target_folder, unique_object + ".jpg")
            image = cv2.imread(image_path)
            height, width = image.shape[0], image.shape[1]
            if (height != image_height) or (width != image_width):
                image = cv2.resize(
                    image, (image_width, image_height), interpolation=cv2.INTER_LINEAR
                )
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
    if not os.path.isdir(raw_train_dataset):
        print(f"Отсутствует {raw_train_dataset}")
        return
    
    if os.path.isdir(train_dataset):
        shutil.rmtree(train_dataset)

    print(f"Копирование {raw_train_dataset} в {train_dataset}")
    shutil.copytree(raw_train_dataset, train_dataset)
    print("Копирование окончено")
    
    folders = os.listdir(train_dataset)

    for folder in folders:
        print(f"Начало предобработки {folder}")
        resize_dataset(
            target_folder=os.path.join(train_dataset, folder),
            image_height=image_height,
            image_width=image_width,
        )
        print(f"Обработка {folder} закончена")


if __name__ == "__main__":
    main()
