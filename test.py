import logging

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
import yaml
import json
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import os

from lib.dataset import MaskDataset
from lib.mobilenet import MobileMaskNet
from lib.my_own_net import MyOwnNet
from lib.wrapper import ModelWrapper
from train_test_config import batch_size, num_workers
from train_test_config import test_df_path as test_paths
from train_test_config import test_weights
from train_test_config import model_type
from lib.tools import get_metrics_3_classes

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

    torch.multiprocessing.set_start_method("forkserver", force=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    result = [{"test_weights": test_weights}]

    criterion = nn.CrossEntropyLoss()
    #metric = accuracy_score
    metric = get_metrics_3_classes
    
    if model_type == "mobile":
        model = MobileMaskNet().to(device)
    else:
        model = MyOwnNet().to(device)
    
    model_wrapper = ModelWrapper(model=model)

    try:
        model_wrapper.load(path_to_model=test_weights)
    except FileNotFoundError as e:
        logger.error(f'Файл с весами модели {test_weights} не найден')
        return
        
    transform_test = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ]
        )

    # Если указаны тестовые датафреймы, то для каждого формируем СВОЙ даталоадер
    try:
        test_dfs = [pd.read_pickle(test_path) for test_path in test_paths]
    except FileNotFoundError as e:
        logger.error(f'Ошибка при загрузке датафреймов {test_paths}')
        return
    
    for test_df in test_dfs:
        test_df["image_paths"] = test_df["image_paths"].apply(
            lambda x: os.path.normpath("./data/" + x)
        )
    
    test_datasets = [MaskDataset(test_df, transform_test) for test_df in test_dfs]
    test_data_loaders = [
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        for test_dataset in test_datasets
    ]
    logger.info(f"Test will be executed for weights: {test_weights}")
    logger.info(f"Test will be executed on {test_paths}")
    test_results = [
        model_wrapper.valid(criterion, metric, test_data_loader, device)
        for test_data_loader in test_data_loaders
    ]
    test_losses = [test_result["valid_loss"] for test_result in test_results]
    test_metrics = [test_result["valid_metric"] for test_result in test_results]
    logger.info(f"Metrics on test datasets: {test_metrics}")
    result.append({"test_path": test_paths})
    result.append({'test_losses': [test_loss.item() for test_loss in test_losses]})
    result.append(
        {"test_metrics": [test_metric for test_metric in test_metrics]}
    )

    with open(r"./test_result.json", "w") as file:
        json.dump(result, file)


if __name__ == "__main__":
    main()
