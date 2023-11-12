import os
import cv2
import json
import pandas as pd
import numpy as np
from json import JSONDecodeError
from typing import List

from config import target_datasets, image_height, image_width


def get_result_dataframe(
    target_datasets: List[str], image_height: int, image_width: int
) -> pd.DataFrame:

    dataframes = [
        get_dataframe_for_dataset(target_dataset, image_height, image_width)
        for target_dataset in target_datasets
    ]
    result_dataframe = pd.concat(dataframes)
    result_dataframe.reset_index(drop=True, inplace=True)
    return result_dataframe


def get_dataframe_for_dataset(
    target_folder: str, image_height: int, image_width: int
) -> pd.DataFrame:

    filenames = os.listdir(target_folder)
    cutted_filenames = [filename.split(".")[0] for filename in filenames]
    unique_objects = list(set(cutted_filenames))

    image_paths = []
    keypoints = []
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

    data = {}
    data["image_paths"] = image_paths
    data["keypoints"] = keypoints
    data = pd.DataFrame.from_dict(data)
    return data


def main():
    result_dataframe = get_result_dataframe(target_datasets, image_height, image_width)
    pd.to_pickle(result_dataframe, "./dataframe.pkl")


if __name__ == "__main__":
    main()
