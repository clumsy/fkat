# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from lightning.pytorch.profilers import Profiler
from unittest import TestCase
from unittest.mock import MagicMock, patch

from torch.utils.data import DataLoader, Dataset

from fkat.data import data_module
from fkat.data.data_module import DataModule, worker_init_fn


class TestWorkerInitFn(TestCase):
    @patch(f"{data_module.__name__}.profile_until_exit")
    def test_worker_init_fn(self, mock_profiler):
        # Arrange
        mock_profiler.return_value = None
        test_profiler = MagicMock()
        test_stage = "train"
        test_worker_id = 10
        test_init_fn = MagicMock()
        test_init_fn.return_value = None
        # Act
        worker_init_fn(test_profiler, test_stage, test_init_fn, test_worker_id)
        # Assert
        mock_profiler.assert_called_once_with(
            test_profiler,
            action=f"DataWorker[{test_stage}][{test_worker_id}]",
            filename_suffix=f"_{test_stage}_{test_worker_id}",
        )

    @patch(f"{data_module.__name__}.profile_until_exit")
    def test_worker_init_fn_diff_worker_id(self, mock_profiler):
        # Arrange
        mock_profiler.return_value = None
        test_profiler = MagicMock()
        test_stage = "train"
        test_worker_id = 42
        test_init_fn = MagicMock()
        test_init_fn.return_value = None
        # Act
        worker_init_fn(test_profiler, test_stage, test_init_fn, test_worker_id)
        # Assert
        mock_profiler.assert_called_once_with(
            test_profiler,
            action=f"DataWorker[{test_stage}][{test_worker_id}]",
            filename_suffix=f"_{test_stage}_{test_worker_id}",
        )


class DataModuleTest(TestCase):
    def test_dataloaders(self):
        dm = DataModule(dataloaders={"test": {"dataset": (dataset := MagicMock(spec=Dataset))}})
        assert dataset == dm.test_dataloader().dataset  # type: ignore[attr-defined]
        assert dm.train_dataloader() is None
        assert dm.predict_dataloader() is None
        assert dm.val_dataloader() is None

    @patch(f"{data_module.__name__}.DataLoader", spec=DataLoader)
    def test_prepare_data(self, mock_dataloader):
        # Arrange
        dm = DataModule(dataloaders={"predict": {"dataset": (dataset := MagicMock(spec=Dataset))}})
        mock_dataloader.return_value.dataset = dataset
        dataset.prepare_data = MagicMock()
        # Act
        dataloader = dm.predict_dataloader()
        dm.prepare_data()
        # Assert
        assert dataloader == mock_dataloader.return_value
        dataset.prepare_data.assert_called()

    @patch(f"{data_module.__name__}.DataLoader", spec=DataLoader)
    def test_setup(self, mock_dataloader):
        # Arrage
        dm = DataModule(dataloaders={"train": {"dataset": (dataset := MagicMock(spec=Dataset))}})
        dm.trainer = MagicMock()
        mock_dataloader.return_value.dataset = dataset
        dataset.set_device = MagicMock()
        dataset.setup = MagicMock()
        # Act
        dataloader = dm.train_dataloader()
        dm.setup(stage := "fit")
        # Assert
        assert dataloader == mock_dataloader.return_value
        dataset.set_device.assert_called_once_with(dm.trainer.strategy.root_device)
        dataset.setup.assert_called_once_with(stage)

    @patch(f"{data_module.__name__}.DataLoader", spec=DataLoader)
    def test_on_exception(self, mock_dataloader):
        # Arrage
        dm = DataModule(dataloaders={"test": {"dataset": (dataset := MagicMock(spec=Dataset))}})
        mock_dataloader.return_value.dataset = dataset
        dataset.on_exception = MagicMock()
        exception = AssertionError()
        # Act
        dataloader = dm.test_dataloader()
        dm.on_exception(exception)
        # Assert
        assert dataloader == mock_dataloader.return_value
        dataset.on_exception.assert_called_once_with(exception)

    @patch(f"{data_module.__name__}.DataLoader", spec=DataLoader)
    def test_teardown(self, mock_dataloader):
        # Arrage
        dm = DataModule(dataloaders={(stage := "test"): {"dataset": (dataset := MagicMock(spec=Dataset))}})
        mock_dataloader.return_value.dataset = dataset
        dataset.teardown = MagicMock()
        # Act
        dataloader = dm.test_dataloader()
        dm.teardown(stage)
        # Assert
        assert dataloader == mock_dataloader.return_value
        dataset.teardown.assert_called_once_with(stage)

    @patch(f"{data_module.__name__}.get_rng_states")
    @patch(f"{data_module.__name__}.DataLoader", spec=DataLoader)
    def test_state_dict(self, mock_dataloader, mock_get_rng_states):
        # Arrange
        mock_get_rng_states.return_value = {"some_rng": "some_value"}
        mock_state_dict = MagicMock()
        mock_state_dict.return_value = {"test_key": "test_val"}
        mock_dataloader.return_value.state_dict = mock_state_dict

        dm = DataModule(dataloaders={"test": mock_dataloader})
        dataloader = dm.test_dataloader()

        # Act
        actual_state_dict = dm.state_dict()
        # Expected state_dict
        expected_state_dict = {"test": {"some_rng": "some_value", "test_key": "test_val"}}

        # Assert
        assert isinstance(actual_state_dict, dict)
        assert isinstance(expected_state_dict, dict)
        assert actual_state_dict == expected_state_dict
        assert dataloader == mock_dataloader.return_value

    @patch(f"{data_module.__name__}.set_rng_states")
    @patch(f"{data_module.__name__}.DataLoader", spec=DataLoader)
    def test_load_state_dict(self, mock_dataloader, mock_set_rng_states):
        # Arrange
        mock_set_rng_states.return_value = {"some_rng": "some_value"}
        mock_load_state_dict = MagicMock()
        mock_load_state_dict.return_value = {"test_key": "test_val"}
        mock_dataloader.return_value.load_state_dict = mock_load_state_dict

        dm = DataModule(dataloaders={"test": mock_dataloader})
        dataloader = dm.test_dataloader()
        test_state_dict = {"test": {"some_rng": "some_value", "test_key": "test_val"}}

        # Act
        dm.load_state_dict(test_state_dict)

        # Assert
        assert dataloader == mock_dataloader.return_value
        dataloader.load_state_dict.assert_called_once_with(test_state_dict["test"])  # type: ignore[attr-defined]

    @patch(f"{data_module.__name__}.DataLoader", spec=DataLoader)
    def test_on_save_checkpoint(self, mock_dataloader):
        # Arrage
        dm = DataModule(dataloaders={"val": {"dataset": (dataset := MagicMock(spec=Dataset))}})
        mock_dataloader.return_value.dataset = dataset
        dataset.on_save_checkpoint = MagicMock()
        # Act
        dataloader = dm.val_dataloader()
        dm.on_save_checkpoint(checkpoint := {"checkpoint": "test"})
        # Assert
        assert dataloader == mock_dataloader.return_value
        dataset.on_save_checkpoint.assert_called_once_with(checkpoint)

    @patch(f"{data_module.__name__}.DataLoader", spec=DataLoader)
    def test_on_load_checkpoint(self, mock_dataloader):
        # Arrage
        dm = DataModule(dataloaders={"test": mock_dataloader})
        mock_dataloader.return_value.dataset = (dataset := MagicMock())
        dataset.on_load_checkpoint = MagicMock()
        # Act
        _ = dm.test_dataloader()
        dm.on_load_checkpoint(checkpoint := {"checkpoint": "test"})
        # Assert
        dataset.on_load_checkpoint.assert_called_once_with(checkpoint)

    def test_instruments_with_profiler(self):
        # Arrange
        profiler = MagicMock(spec=Profiler)
        profiler.filename = (filename := "some_filename")
        dm = DataModule(
            profiler=profiler,
            dataloaders={
                (stage := "test"): {
                    "dataset": MagicMock(spec=Dataset),
                    "worker_init_fn": (worker_init_fn := MagicMock()),
                }
            },
        )
        # Act
        dm.test_dataloader().worker_init_fn(worker_id := 42)  # type: ignore[attr-defined]
        # Assert
        worker_init_fn.assert_called_once_with(worker_id)
        assert profiler.filename == f"{filename}_{stage}_{worker_id}"
