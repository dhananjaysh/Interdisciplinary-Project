from abc import ABCMeta, abstractmethod
import torch


class PerformanceMeasure(metaclass=ABCMeta):
    """
    A performance measure.
    """

    @abstractmethod
    def reset(self):
        """
        Resets internal state.
        """

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        """

        pass

    @abstractmethod
    def __str__(self) -> str:
        """
        Return a string representation of the performance.
        """

        pass


class Accuracy(PerformanceMeasure):
    """
    Average classification accuracy.
    """

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        """
        Resets the internal state.
        """
        self.correct_pred = {classname: 0 for classname in self.classes}
        self.total_pred = {classname: 0 for classname in self.classes}
        self.n_matching = 0  # number of correct predictions
        self.n_total = 0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (batchsize,n_classes) with each row being a class-score vector.
        target must have shape (batchsize,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        [len(prediction.shape) should be equal to 2, and len(target.shape) should be equal to 1.]
        """

        if prediction.ndim != 2 or target.ndim != 1:
            raise ValueError("Prediction must be 2D and target must be 1D.")
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Batch size mismatch between prediction and target.")

        pred_labels = torch.argmax(prediction, dim=1)

        self.n_total += target.size(0)
        self.n_matching += (pred_labels == target).sum().item()

        for pred, true in zip(pred_labels, target):
            class_name = self.classes[true.item()]
            self.total_pred[class_name] += 1
            if pred.item() == true.item():
                self.correct_pred[class_name] += 1

    def __str__(self):
        """
        Return a string representation of the performance, accuracy and per class accuracy.
        """
        overall_acc = self.accuracy()
        per_class_acc = self.per_class_accuracy()
        
        result = [
            f"accuracy: {overall_acc:.4f}",
            f"per class accuracy: {per_class_acc:.2f}"
        ]

        for cls in self.classes:
            total = self.total_pred[cls]
            correct = self.correct_pred[cls]
            acc = correct / total if total > 0 else 0.0
            result.append(f"Accuracy for class: {cls:<6} is {acc:.2f}")

        return "\n".join(result)

    def accuracy(self) -> float:
        """
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """

        if self.n_total == 0:
            return 0.0
        return self.n_matching / self.n_total

    def per_class_accuracy(self) -> float:
        """
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        """
        class_accuracies = []
        for cls in self.classes:
            total = self.total_pred[cls]
            correct = self.correct_pred[cls]
            if total > 0:
                class_accuracies.append(correct / total)

        if len(class_accuracies) == 0:
            return 0.0

        return sum(class_accuracies) / len(class_accuracies)
