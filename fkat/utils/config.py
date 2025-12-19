# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from typing import Any, Protocol
from tempfile import TemporaryDirectory

from lightning import Trainer as LightningTrainer
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

from fkat.utils.mlflow import mlflow_logger


class Trainer(Protocol):
    """
    Protocol defining the interface for training models.

    This protocol establishes the required methods that any trainer implementation
    must provide. It serves as a structural subtyping interface for objects that
    can train, evaluate, and make predictions with machine learning models.

    Implementations of this protocol should handle the complete training lifecycle,
    including model fitting, prediction, testing, and validation.

    Note:
        As a Protocol class, this is not meant to be instantiated directly but
        rather used for type checking and interface definition.
    """

    def fit(self, *args: Any, **kwargs: Any) -> Any:
        """
        Train a model on the provided data.

        This method handles the model training process, including data loading,
        optimization, and potentially checkpointing.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Training results, which may include training history,
                 trained model, or other relevant information.
        """
        ...

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """
        Generate predictions using a trained model.

        This method applies the trained model to new data to produce predictions.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Model predictions, which could be probabilities, class labels,
                 regression values, or other outputs depending on the model type.
        """
        ...

    def test(self, *args: Any, **kwargs: Any) -> Any:
        """
        Evaluate a trained model on test data.

        This method assesses model performance on a test dataset, typically
        calculating metrics such as accuracy, loss, or other relevant measures.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Test results, which may include metrics, predictions,
                 or other evaluation information.
        """
        ...

    def validate(self, *args: Any, **kwargs: Any) -> Any:
        """
        Evaluate a trained model on validation data.

        This method assesses model performance on a validation dataset, which is
        typically used during the training process for hyperparameter tuning
        or early stopping.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: Validation results, which may include metrics, predictions,
                 or other evaluation information.
        """
        ...


class SingletonResolver:
    """
    A singleton class that resolves and manages training components.

    This class serves as a central registry for training-related objects such as
    trainers, data, models, and checkpoint paths. It ensures that these components
    are accessible throughout the application while maintaining a single instance.

    Example:
        >>> resolver = SingletonResolver()
        >>> resolver.trainer = MyTrainer()
        >>> resolver.model = MyModel()
        >>> # Access the same instance elsewhere
        >>> same_resolver = SingletonResolver()
        >>> assert same_resolver.trainer is resolver.trainer
    """

    trainer: Trainer
    """The trainer instance responsible for executing the training process."""

    data: Any | None = None
    """The dataset or data loader used for training and evaluation.
    Defaults to None."""

    model: Any | None = None
    """The model architecture to be trained or evaluated.
    Defaults to None."""

    ckpt_path: Any | None = None
    """Path to checkpoint files for model loading/saving.
    Defaults to None."""

    return_predictions: Any | None = None
    """Flag or configuration for returning predictions.
    Defaults to None."""

    tuners: Any | None = None
    """Hyperparameter tuners or optimization components.
    Defaults to None."""


def register_singleton_resolver() -> Any:
    resolver = SingletonResolver()

    def resolve(key: str) -> Any:
        res = resolver
        for attr in key.split("."):
            res = getattr(res, attr)
        return res

    OmegaConf.register_new_resolver("fkat", resolve)
    return resolver


def to_str(cfg: Any) -> str:
    """
    Convert a configuration object to a formatted string representation.

    This function takes a configuration object and converts it to a human-readable
    YAML string. It's useful for logging, debugging, or displaying configuration settings.

    Args:
        cfg (Any): The configuration object to convert to string.

    Returns:
        str: A formatted string representation of the configuration.

    Example:
        >>> config = {"model": {"type": "resnet", "layers": 50}, "batch_size": 32}
        >>> print(to_str(config))
        Config:
        model:
          type: resnet
          layers: 50
        batch_size: 32
    """
    return "Config:\n" + OmegaConf.to_yaml(cfg)


def to_primitive_container(cfg: Any) -> Any:
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg)
    return cfg


@rank_zero_only
def save(cfg: Any, trainer: LightningTrainer) -> None:
    yaml_str = OmegaConf.to_yaml(cfg)
    with TemporaryDirectory() as temp_dir:
        yaml_path = os.path.join(temp_dir, "config.yaml")
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, "w") as f:
            f.write(yaml_str)
        if mlflow := mlflow_logger(trainer):
            if mlflow.run_id:
                mlflow.experiment.log_artifact(mlflow.run_id, yaml_path)
