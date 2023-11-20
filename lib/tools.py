import json
import os
from json import JSONDecodeError
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .exceptions import InvalidDatasetStructure

def get_result_dataframe(
    target_dataset: str,
    image_height: int,
    image_width: int,
    target_mapper: Optional[Dict[str, int]],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    
    object_folders = os.listdir(target_dataset)
    object_paths = [
        os.path.join(target_dataset, object_folder)
        for object_folder in object_folders
        ]
    
    if target_mapper:
        try:
            for object_folder in object_folders:
                target_mapper[object_folder]
        except KeyError:
            raise InvalidDatasetStructure
            
        if len(object_folders) != len(target_mapper):
            raise InvalidDatasetStructure
    else:
        target_mapper = {object_folders[i]: i for i in range(len(object_folders))}

    dataframes = [
        get_dataframe_for_dataset(object_path, image_height, image_width, target_mapper)
        for object_path in object_paths
    ]
    result_dataframe = pd.concat(dataframes)
    result_dataframe.reset_index(drop=True, inplace=True)
    return result_dataframe, target_mapper


def get_dataframe_for_dataset(
    target_folder: str, image_height: int, image_width: int, target_mapper: Dict
) -> pd.DataFrame:

    filenames = os.listdir(target_folder)
    cutted_filenames = [filename.split(".")[0] for filename in filenames]
    unique_objects = list(set(cutted_filenames))

    image_paths = []
    keypoints = []
    dataset_idxs = []
    for unique_object in unique_objects:
        try:
            image_path = os.path.join(target_folder, unique_object + ".jpg")
            cv2.imread(image_path)

            keypoint_path = os.path.join(target_folder, unique_object + ".json")
            with open(keypoint_path, "r") as f:
                keypoint = json.load(f)

            x = keypoint[0]["x"]
            y = keypoint[0]["y"]

            if (x < 0) or (x > 1) or (y < 0) or (y > 1):
                print(f"Ошибка координат {keypoint}")
                continue

            x = int(x * image_width)
            y = int(y * image_height)

            image_paths.append(image_path)
            keypoints.append(np.array([x, y]))
            dataset_idxs.append(target_mapper[target_folder.split("/")[-1]])

        except JSONDecodeError:
            print(f"Ошибка формата JSON {keypoint_path}")
            continue

        except FileNotFoundError as e:
            print(e)
            continue

        except IndexError:
            print(f"Ошибка формата JSON {keypoint_path}")
            continue

        except Exception as e:
            print(keypoint)
            print(e)
            continue

    return pd.DataFrame.from_dict(
        {
            "image_paths": image_paths,
            "keypoints": keypoints,
            "dataset_idxs": dataset_idxs,
        }
    )
