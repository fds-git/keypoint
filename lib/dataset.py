from torch.utils.data import Dataset
import pandas as pd
import cv2
import numpy as np
import torch
from typing import Callable


class KeypointDataset(Dataset):
    """Класс для создания датасетов"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Callable,
        image_width: int,
        image_height: int,
        horizontal_flip: bool = False,
    ):

        self.image_paths = dataframe["image_paths"]
        self.keypoints = dataframe["keypoints"]
        self.dataset_idxs = dataframe["dataset_idxs"]
        self.transform = transform
        self.data_len = len(dataframe.index)
        self.horizontal_flip = horizontal_flip
        self.image_width = image_width
        self.image_height = image_height

    def __getitem__(self, index: int):

        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype("float")
        height, width = image.shape[0], image.shape[1]

        if (height != self.image_height) or (width != self.image_width):
            image = cv2.resize(
                image,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_LINEAR,
            )

        keypoints = self.keypoints[index]
        transformed = self.transform(image=image, keypoints=[keypoints])
        transformed_image = transformed["image"].float()
        transformed_keypoints = transformed["keypoints"]
        transformed_keypoints = torch.FloatTensor(
            np.array(transformed_keypoints)
        ).float()
        transformed_keypoints = transformed_keypoints.view(-1)

        return transformed_image, self.dataset_idxs[index], transformed_keypoints

    def __len__(self):
        return self.data_len
