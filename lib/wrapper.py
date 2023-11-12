import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import time
import numpy as np
import logging
from typing import Callable, Dict, Tuple
from tqdm import tqdm

logger = logging.getLogger("main")


class ModelWrapper(nn.Module):
    """Класс, реализующий функционал для работы с нейронной сетью"""

    def __init__(self, model: object):
        """Конструктор класса
        Входные параметры:
        model: object - последовательность слоев или модель,
        через которую будут проходить данные"""

        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input_data: torch.Tensor):
        """Метод прямого прохода через объект класса
        Входные параметры:
        input_data: torch.Tensor - тензорное представление входа нейронной сети
        Возвращаемые значения:
        output_data: torch.Tensor - тензорное представление выхода нейронной сети"""

        output_data = self.model(input_data)
        return output_data

    def fit(
        self,
        criterion: Callable,
        metrics: Dict[str, Callable],
        optimizer: object,
        scheduler: object,
        train_data_loader: DataLoader,
        epochs: int,
        early_stopping: int,
        directory_to_save: str,
        device: str,
        valid_data_loader: DataLoader = None,
        verbose: int = 5,
    ):
        """Метод для обучения модели
        Входные параметры:
        criterion: Callable - объект для вычисления loss
        metric: Callable - объект для вычисления метрики качества
        optimizer: object - оптимизатор
        scheduler: object - объект для изменения lr в ходе обучения
        train_data_loader: DataLoader - загрузчик данных для обучения
        valid_data_loader: DataLoader - загрузчик данных для валидации
        epochs: int - количество эпох обучения
        early_stopping: int - количество эпох без улучшения метрики при
        валидации для ранней остановки
        (только если valid_data_loader != None)
        verbose: int - вывод информации через каждые verbose итераций
        directory_to_save: str - директория для сохранения модели после каждой эпохи
        device: str - устройство выполнения операций
        Возвращаемые значения:
        result: dict - словарь со значениями loss при тренировке, валидации и
        метрики при валидации для каждой эпохи"""

        best_valid_loss = 999999
        num_epochs_without_improve = 0

        epoch_train_losses = []
        epoch_valid_losses = []
        epoch_valid_metrics = []
        result = {}

        for epoch in range(epochs):
            self.model.train()
            time1 = time.time()
            running_loss = 0.0
            train_losses = []

            logger.info("=" * 35 + f" Epoch: {epoch+1} " + "=" * 35)

            for batch_idx, data in enumerate(train_data_loader):
                images = data[0].to(device)
                dataset_idxs = data[1].to(device)
                target = data[2].to(device)
                optimizer.zero_grad()

                outputs = self.model(images, dataset_idxs)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                train_losses.append(loss.item())
                if (batch_idx + 1) % verbose == 0:
                    logger.info(
                        f"Batch {batch_idx+1}/{len(train_data_loader)}, "
                        + f"Loss: {(running_loss/verbose):.4f}"
                    )

                    running_loss = 0.0

            logger.info("=" * 80)
            time2 = time.time()
            logger.info(f"Epoch {epoch+1}")
            logger.info(f"Time: {(time2-time1):.2f} sec")
            logger.info(f"Lr: {scheduler.get_last_lr()[0]:.5f}")

            train_loss = np.mean(train_losses)
            epoch_train_losses.append(train_loss)

            scheduler.step()

            if valid_data_loader is not None:
                valid_result = self.valid(criterion, metrics, valid_data_loader, device)
                valid_loss = valid_result["valid_loss"]
                valid_metric = valid_result["valid_metric"]

                epoch_valid_losses.append(valid_loss)
                epoch_valid_metrics.append(valid_metric)

                self.save(
                    path_to_save=f"{directory_to_save}/ep_{epoch+1}_valid_loss_{valid_loss:.4f}"
                )
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    num_epochs_without_improve = 0
                else:
                    num_epochs_without_improve += 1

                logger.info(f"Train loss: {(train_loss):.3f}")
                logger.info(f"Valid loss: {(valid_loss):.3f}")
                logger.info(f"Valid metrics: {(valid_metric)}")

                if num_epochs_without_improve == early_stopping:
                    logger.info(
                        f"{num_epochs_without_improve} epochs without"
                        + "loss decreasing. Training will be stopped"
                    )
                    break
            else:
                logger.info(f"Train loss: {(train_loss):.3f}")
                epoch_valid_losses.append(None)
                epoch_valid_metrics.append(None)

                self.save(
                    path_to_save=f"{directory_to_save}/ep_{epoch+1}_train_loss_{train_loss:.4f}"
                )

        result["epoch_train_losses"] = epoch_train_losses
        result["epoch_valid_losses"] = epoch_valid_losses
        result["epoch_valid_metrics"] = epoch_valid_metrics

        return result

    def valid(
        self,
        criterion: Callable,
        metrics: Dict[str, Callable],
        valid_data_loader: DataLoader,
        device: str,
    ):
        """Метод для валидации модели
        Входные параметры:
        criterion: object - объект для вычисления loss
        metric: object - объект для вычисления метрики качества
        valid_data_loader: DataLoader - загрузчик данных для валидации
        device: str - устройство для выполнения операций
        Возвращаемые значения:
        result: dict - словарь со значениями loss и метрики при валидации"""

        self.model.eval()
        result = {}
        all_true = []
        all_pred = []

        with torch.no_grad():
            for data in tqdm(valid_data_loader):
                images = data[0].to(device)
                dataset_idxs = data[1].to(device)
                target = data[2].to(device)

                all_true.append(target)
                pred = self.model(images, dataset_idxs)
                all_pred.append(pred)

        all_true = torch.vstack(all_true)
        all_pred = torch.vstack(all_pred)

        result["valid_loss"] = criterion(all_pred, all_true).item()
        result["valid_metric"] = {
            key: round(value(all_pred, all_true), 3) for key, value in metrics.items()
        }
        self.model.train()

        return result

    def predict(self, predict_data_loader: DataLoader, device: str):
        """Метод получения предсказания модели
        Входные параметры:
        predict_data_loader: DataLoader - загрузчик данных для валидации
        device: str - устройство выполнения операций
        Возвращаемые значения:
        result: np.ndarray - предсказания модели"""

        self.model.eval()
        result = []
        with torch.no_grad():
            for batch_idx, data in enumerate(predict_data_loader):
                images = data[0].to(device)
                dataset_idxs = data[1].to(device)
                outputs = self.model(images, dataset_idxs)
                landmarks = outputs.view(images.shape[0], 2)
                landmarks = landmarks.detach().cpu().numpy()
                result.append(landmarks)
        self.model.train()

        result = np.vstack(result)
        return result

    def save(self, path_to_save: str):
        """Метод сохранения весов модели
        Входные параметры:
        path_to_save: str - директория для сохранения состояния модели"""

        torch.save(self.model.state_dict(), path_to_save)

    def trace_save(self, path_to_save: str, example_forward_input: Tuple[torch.tensor, torch.tensor]):
        """Метод сохранения модели через torchscript
        Входные параметры:
        path_to_save: str - директория для сохранения модели
        example_forward_input: torch.tensor - тензор для трассировки"""

        traced_model = torch.jit.trace((self.model).eval(), example_forward_input)
        torch.jit.save(traced_model, path_to_save)

    def save_to_onnx(self, path_to_save: str, example_forward_input: torch.tensor):
        """Метод конвертации модели в ONNX формат
        Входные параметры:
        path_to_save: str - директория для сохранения модели
        example_forward_input: torch.tensor - тензор для трассировки"""

        torch.onnx.export(
            self.model.to("cpu"),
            example_forward_input,
            path_to_save,
            # store the trained parameter weights inside the model file
            export_params=True,
            # the ONNX version to export the model to
            # opset_version=10,
            # whether to execute constant folding for optimization
            do_constant_folding=True,
            # the model's input names
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )  # the model's output names

    def load(self, path_to_model: str):
        """Метод загрузки весов модели
        Входные параметры:
        path_to_model: str - директория с сохраненными весами модели"""

        self.model.load_state_dict(torch.load(path_to_model))
