import logging
import time
import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from lib.dataset import KeypointDataset
from lib.mobilenet import MobileKeypointNet
from lib.wrapper import ModelWrapper
from config import batch_size, num_workers, test_df_path
from config import test_weights
from config import image_height, image_width, treashold
from lib.metrics import MeanRelativeDistance, MeanAccuracy

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

    model = MobileKeypointNet().to(device)
    model_wrapper = ModelWrapper(model=model)

    try:
        model_wrapper.load(path_to_model=test_weights)
    except FileNotFoundError:
        logger.error(f"Файл с весами модели {test_weights} не найден")
        return

    transform_test = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    # Если указаны тестовые датафреймы, то для каждого формируем СВОЙ даталоадер
    try:
        test_df = pd.read_pickle(test_df_path)
    except FileNotFoundError:
        logger.error(f"Ошибка при загрузке датафрейма {test_df_path}")
        return

    test_dataset = KeypointDataset(
        dataframe=test_df,
        transform=transform_test,
        image_width=image_width,
        image_height=image_height,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    logger.info(f"Test will be executed for weights: {test_weights}")
    logger.info(f"Test will be executed on {test_df_path}")

    start = time.time()
    test_result = model_wrapper.valid(criterion, metrics, test_data_loader, device)
    stop = time.time()

    test_loss = test_result["valid_loss"]
    test_metric = test_result["valid_metric"]
    logger.info(f"Metrics on test datasets: {test_metric}")
    logger.info(f"Total time: {stop - start} seconds")

    result.append({"test_path": test_df_path})
    result.append({"test_loss": test_loss})
    result.append({"test_metric": test_metric})


if __name__ == "__main__":
    main()
