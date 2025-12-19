# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch, MagicMock

from omegaconf import Resolver, DictConfig

from fkat import setup


class TestSetup:
    @patch.dict("os.environ", {}, clear=True)
    @patch("torch.backends.cudnn")
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.use_deterministic_algorithms")
    @patch("fkat.pdb.post_mortem")
    @patch("fkat.config")
    @patch("lightning.seed_everything")
    @patch("torch.multiprocessing.set_start_method")
    @patch("omegaconf.OmegaConf")
    def test_setup(
        self,
        mock_oc,
        mock_mp,
        mock_seed,
        mock_config,
        mock_post_mortem,
        mock_det_algo,
        mock_cuda,
        mock_cudnn,
    ):
        """Test setup function with all possible configurations."""

        # Test empty config
        cfg = DictConfig({})
        setup(cfg)
        mock_mp.assert_called_with("spawn", force=True)
        mock_config.register_singleton_resolver.assert_called_once()

        # Reset mocks
        mock_mp.reset_mock()
        mock_config.reset_mock()

        # Test full config
        cfg = DictConfig(
            {
                "setup": {
                    "multiprocessing": "fork",
                    "print_config": True,
                    "seed": 42,
                    "post_mortem": True,
                    "determinism": True,
                }
            }
        )

        setup(cfg, **cfg["setup"], resolvers={"myresolver": (resolver := MagicMock(spec=Resolver))})

        # Verify all calls
        mock_mp.assert_called_once_with("fork", force=True)
        mock_config.to_str.assert_called_once_with(cfg)
        mock_seed.assert_called_once_with(42)
        mock_post_mortem.assert_called_once()
        mock_det_algo.assert_called_once_with(True)
        assert mock_cudnn.deterministic
        assert not mock_cudnn.benchmark
        import os

        assert os.getenv("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO") == "0"
        mock_config.register_singleton_resolver.assert_called_once()

        # Reset mocks
        mock_mp.reset_mock()
        mock_config.to_str.reset_mock()
        mock_seed.reset_mock()
        mock_post_mortem.reset_mock()
        mock_det_algo.reset_mock()
        mock_config.reset_mock()
        os.environ.clear()

        # Test determinism without CUDA
        mock_cuda.return_value = False
        cfg = DictConfig(
            {
                "setup": {
                    "seed": 42,
                    "determinism": True,
                }
            }
        )

        setup(cfg, **cfg["setup"])

        mock_det_algo.assert_called_once_with(True)
        assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ

        mock_oc.register_new_resolver.assert_called_once_with("myresolver", resolver, replace=True)
