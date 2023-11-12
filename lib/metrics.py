import torch


class MeanRelativeDistance:
    def __init__(self, image_width: int, image_height: int):

        super(MeanRelativeDistance, self).__init__()
        self.image_width = image_width
        self.image_height = image_height

    def __call__(self, pred: torch.Tensor, true: torch.Tensor):
        true = true.clone()
        pred = pred.clone()

        true[:, 0] = true[:, 0] / float(self.image_width)
        true[:, 1] = true[:, 1] / float(self.image_height)

        pred[:, 0] = pred[:, 0] / float(self.image_width)
        pred[:, 1] = pred[:, 1] / float(self.image_height)

        mean_relative_distance = torch.mean(
            torch.sqrt(torch.sum(torch.pow(torch.subtract(true, pred), 2), dim=1))
        )

        return mean_relative_distance.item()


class MeanAccuracy:
    def __init__(self, image_width: int, image_height: int, treashold: float):

        super(MeanAccuracy, self).__init__()
        self.image_width = image_width
        self.image_height = image_height
        self.treashold = treashold

    def __call__(self, pred: torch.Tensor, true: torch.Tensor):
        true = true.clone()
        pred = pred.clone()

        true[:, 0] = true[:, 0] / float(self.image_width)
        true[:, 1] = true[:, 1] / float(self.image_height)

        pred[:, 0] = pred[:, 0] / float(self.image_width)
        pred[:, 1] = pred[:, 1] / float(self.image_height)

        relative_distances = torch.sqrt(
            torch.sum(torch.pow(torch.subtract(true, pred), 2), dim=1)
        )
        accuracy = torch.sum(relative_distances < self.treashold) / len(
            relative_distances
        )

        return accuracy.item()