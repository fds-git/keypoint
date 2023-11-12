import logging
import os
import shutil
from datetime import datetime
from sklearn.model_selection import train_test_split

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from lib.dataset import KeypointDataset
from lib.mobilenet import MobileKeypointNet
from lib.wrapper import ModelWrapper
from lib.metrics import MeanRelativeDistance, MeanAccuracy
from config import (
    batch_size,
    early_stopping,
    gamma,
    learning_rate,
    num_epochs,
    num_workers,
    rotate_limit,
    train_df_path,
    full_data
)
from config import verbose, image_height, image_width, treashold

logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
# create console handler with a higher log level
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
# create formatter and add it to the handler
format = (
    "%(asctime)s [%(process)d] [%(name)s]"
    + "%(filename)s:%(lineno)d [%(levelname)s] - %(message)s"
)
formatter = logging.Formatter(format)
handler.setFormatter(formatter)
# add the handler to the logger
logger.addHandler(handler)


def main():

    # Чтобы num_workers в DataLoader работали
    # torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_start_method("forkserver", force=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Узнаем сколько экспериментов было проведено в корневой папке
    file_names = os.listdir("./experiments/")
    number_of_experiments = len(file_names)
    new_experimens_number = number_of_experiments + 1

    # Узнаем текущую дату и время
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")

    # Создаем папку, в которой будет храниться информация об эксперименте
    exp_path = f"./experiments/exp_{new_experimens_number}_{dt_string}"
    weights_path = f"{exp_path}/weights/"
    os.mkdir(exp_path)
    os.mkdir(weights_path)

    # Создаем файл с логами
    logging.basicConfig(filename=f"{exp_path}/logs.log", level=logging.INFO)
    # Сохраняем конфиги в папку эксперимента
    shutil.copyfile("./config.py", f"{exp_path}/config.py")

    transform_train = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.Rotate(
                limit=rotate_limit,
                interpolation=1,
                border_mode=4,
                value=None,
                always_apply=True,
                p=1,
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False, angle_in_degrees=True)
    )

    transform_test = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    dataframe = pd.read_pickle(train_df_path)
    
    train_df, valid_df = train_test_split(dataframe, test_size=0.25, random_state=42)
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    train_dataset = KeypointDataset(
        train_df, transform_train, image_height=image_height, image_width=image_width
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    valid_dataset = KeypointDataset(
        valid_df, transform_test, image_height=image_height, image_width=image_width
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model = MobileKeypointNet().to(device)
    model_wrapper = ModelWrapper(model=model)

    optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.MSELoss()
    mean_relative_distance = MeanRelativeDistance(
        image_width=image_width, image_height=image_height
    )
    mean_accuracy = MeanAccuracy(
        image_width=image_width, image_height=image_height, treashold=treashold
    )
    metrics = {
        "mean_accuracy": mean_accuracy,
        "mean_relative_distance": mean_relative_distance,
    }

    _ = model_wrapper.fit(
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        train_data_loader=train_data_loader,
        valid_data_loader=valid_data_loader,
        epochs=num_epochs,
        early_stopping=early_stopping,
        verbose=verbose,
        directory_to_save=weights_path,
        device=device,
    )


if __name__ == "__main__":
    main()
