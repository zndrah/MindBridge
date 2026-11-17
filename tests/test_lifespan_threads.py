"""
Tests for MindBridge 2.0 CPU Inference Optimization in FastAPI Lifespan
=======================================================================

Verifies:
1. Successful PyTorch thread configuration (set_num_threads=1, set_num_interop_threads=1).
2. Active intra-op and inter-op thread counts are properly logged.
3. Graceful containment when setting inter-op threads fails (e.g. parallel work already started).
4. Graceful containment when setting intra-op threads fails.
5. Graceful containment when PyTorch import fails.
6. Lifespan startup does not fail or halt when PyTorch tuning fails.
7. Optimization is not falsely claimed when tuning fails.
8. Tests run hermetically without requiring production model downloads.
"""

import builtins
import logging
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.api.app import create_app, lifespan
from backend.api.dependencies import app_state


class TestLifespanCPUOptimization(unittest.IsolatedAsyncioTestCase):
    """Unit tests for CPU inference thread configuration in FastAPI lifespan."""

    async def test_successful_thread_configuration(self):
        """
        Verify lifespan sets intra-op and inter-op threads to 1 and logs active thread counts.
        """
        app = create_app()

        with patch("backend.api.app.app_state.initialize_sync") as mock_init, \
             patch("backend.api.app.app_state.shutdown") as mock_shutdown, \
             patch("torch.set_num_threads") as mock_set_intra, \
             patch("torch.set_num_interop_threads") as mock_set_inter, \
             patch("torch.get_num_threads", return_value=1) as mock_get_intra, \
             patch("torch.get_num_interop_threads", return_value=1) as mock_get_inter, \
             patch("backend.api.app.logger") as mock_logger:

            async with lifespan(app):
                # Verify PyTorch thread configuration was called with 1
                mock_set_intra.assert_called_once_with(1)
                mock_set_inter.assert_called_once_with(1)
                mock_get_intra.assert_called()
                mock_get_inter.assert_called()

                # Verify application initialization proceeded
                mock_init.assert_called_once()

                # Verify successful optimization log with thread counts
                info_formatted = [
                    (call.args[0] % call.args[1:]) if len(call.args) > 1 else str(call.args[0])
                    for call in mock_logger.info.call_args_list
                ]
                configured_logged = any(
                    "PyTorch CPU inference threads configured: intra_op=1, inter_op=1" in msg
                    for msg in info_formatted
                )
                self.assertTrue(configured_logged, f"Expected thread configuration log not found in: {info_formatted}")

                # Ensure no warning was logged
                warning_messages = [str(call.args) for call in mock_logger.warning.call_args_list]
                self.assertFalse(
                    any("Failed to configure PyTorch CPU thread" in msg for msg in warning_messages),
                    f"Unexpected warning logged: {warning_messages}",
                )

            # Lifespan teardown must invoke app_state.shutdown
            mock_shutdown.assert_called_once()

    async def test_graceful_handling_of_interop_thread_failure(self):
        """
        If setting inter-op threads fails (e.g., RuntimeError after parallel work started),
        a sanitized warning is logged, the optimization is NOT claimed, and startup proceeds.
        """
        app = create_app()
        simulated_error = RuntimeError(
            "Error: cannot set number of interop threads after parallel work has started or set_num_interop_threads called"
        )

        with patch("backend.api.app.app_state.initialize_sync") as mock_init, \
             patch("backend.api.app.app_state.shutdown") as mock_shutdown, \
             patch("torch.set_num_threads") as mock_set_intra, \
             patch("torch.set_num_interop_threads", side_effect=simulated_error), \
             patch("backend.api.app.logger") as mock_logger:

            # Lifespan must NOT raise an exception
            async with lifespan(app):
                # App startup must still proceed
                mock_init.assert_called_once()

                # Verify sanitized warning was logged
                warning_messages = [str(call.args) for call in mock_logger.warning.call_args_list]
                warning_logged = any(
                    "Failed to configure PyTorch CPU thread count" in msg and "RuntimeError" in msg
                    for msg in warning_messages
                )
                self.assertTrue(warning_logged, f"Sanitized warning not found in: {warning_messages}")

                # Verify optimization is NOT falsely claimed
                info_messages = [str(call.args) for call in mock_logger.info.call_args_list]
                self.assertFalse(
                    any("PyTorch CPU inference threads configured" in msg for msg in info_messages),
                    "Optimization should not be claimed when inter-op thread tuning fails",
                )

            mock_shutdown.assert_called_once()

    async def test_graceful_handling_of_intra_thread_failure(self):
        """
        If setting intra-op threads fails (e.g., ValueError or RuntimeError),
        a sanitized warning is logged, optimization is NOT claimed, and startup proceeds.
        """
        app = create_app()
        simulated_error = ValueError("Expected a positive integer thread count, got -1")

        with patch("backend.api.app.app_state.initialize_sync") as mock_init, \
             patch("torch.set_num_threads", side_effect=simulated_error), \
             patch("torch.set_num_interop_threads") as mock_set_inter, \
             patch("backend.api.app.logger") as mock_logger:

            async with lifespan(app):
                mock_init.assert_called_once()
                # Inter-op should not be reached if intra-op throws
                mock_set_inter.assert_not_called()

                warning_messages = [str(call.args) for call in mock_logger.warning.call_args_list]
                warning_logged = any(
                    "Failed to configure PyTorch CPU thread count" in msg and "ValueError" in msg
                    for msg in warning_messages
                )
                self.assertTrue(warning_logged, f"Sanitized warning not found in: {warning_messages}")

                info_messages = [str(call.args) for call in mock_logger.info.call_args_list]
                self.assertFalse(
                    any("PyTorch CPU inference threads configured" in msg for msg in info_messages),
                    "Optimization should not be claimed when intra-op thread tuning fails",
                )

    async def test_graceful_handling_of_torch_import_failure(self):
        """
        If PyTorch cannot be imported (ImportError), a sanitized warning is logged,
        optimization is NOT claimed, and startup proceeds.
        """
        app = create_app()
        real_import = builtins.__import__

        def failing_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("No module named 'torch'")
            return real_import(name, *args, **kwargs)

        with patch("backend.api.app.app_state.initialize_sync") as mock_init, \
             patch("builtins.__import__", side_effect=failing_import), \
             patch("backend.api.app.logger") as mock_logger:

            async with lifespan(app):
                mock_init.assert_called_once()

                warning_messages = [str(call.args) for call in mock_logger.warning.call_args_list]
                warning_logged = any(
                    "Failed to configure PyTorch CPU thread count" in msg and "ImportError" in msg
                    for msg in warning_messages
                )
                self.assertTrue(warning_logged, f"Sanitized warning not found in: {warning_messages}")

                info_messages = [str(call.args) for call in mock_logger.info.call_args_list]
                self.assertFalse(
                    any("PyTorch CPU inference threads configured" in msg for msg in info_messages),
                    "Optimization should not be claimed when torch import fails",
                )

    async def test_graceful_handling_of_generic_exception(self):
        """
        Any unexpected exception during thread configuration is caught, logged safely,
        and does not prevent startup.
        """
        app = create_app()

        with patch("backend.api.app.app_state.initialize_sync") as mock_init, \
             patch("torch.set_num_threads", side_effect=OSError("Thread subsystem unavailable")), \
             patch("backend.api.app.logger") as mock_logger:

            async with lifespan(app):
                mock_init.assert_called_once()

                warning_messages = [str(call.args) for call in mock_logger.warning.call_args_list]
                self.assertTrue(
                    any("Failed to configure PyTorch CPU thread count" in msg and "OSError" in msg for msg in warning_messages),
                    f"Expected OSError warning in: {warning_messages}",
                )

                info_messages = [str(call.args) for call in mock_logger.info.call_args_list]
                self.assertFalse(
                    any("PyTorch CPU inference threads configured" in msg for msg in info_messages),
                )

    def test_fastapi_lifespan_integration_without_model_downloads(self):
        """
        FastAPI TestClient lifecycle integration runs cleanly with mocked model initialization,
        confirming no remote or local model files are downloaded or required.
        """
        app = create_app()

        with patch("backend.api.app.app_state.initialize_sync") as mock_init, \
             patch("backend.api.app.app_state.shutdown") as mock_shutdown:

            with TestClient(app) as client:
                res = client.get("/health")
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["status"], "healthy")

            mock_init.assert_called_once()
            mock_shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
