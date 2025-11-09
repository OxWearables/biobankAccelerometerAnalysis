"""
Pytest configuration and fixtures for accelerometer tests.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from tests.test_data_generator import (
    AccelerometerDataGenerator,
    generate_test_dataset
)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="accel_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="session")
def data_generator():
    """Provide a data generator instance with fixed seed."""
    return AccelerometerDataGenerator(seed=42)


@pytest.fixture
def simple_stationary_data(data_generator):
    """Generate simple stationary data for calibration tests."""
    return data_generator.generate_stationary(
        n_samples=10000,
        orientation=(0.0, -1.0, 0.0),
        noise_std=0.005,
        temperature=20.0
    )


@pytest.fixture
def stationary_with_error(data_generator):
    """Generate stationary data with known calibration error."""
    data = data_generator.generate_stationary(
        n_samples=10000,
        orientation=(0.0, -1.0, 0.0),
        noise_std=0.005,
        temperature=20.0
    )
    # Add known calibration error
    calibration_error = {
        'offset': (0.01, -0.02, 0.015),
        'slope': (0.98, 1.02, 0.99),
        'temp_coeff': (0.0001, -0.0002, 0.00015)
    }
    data = data_generator.add_calibration_error(data, **calibration_error)
    return data, calibration_error


@pytest.fixture
def walking_data(data_generator):
    """Generate walking pattern data."""
    return data_generator.generate_walking(
        n_samples=30000,  # 300 seconds at 100Hz
        step_freq=2.0,
        intensity=0.3
    )


@pytest.fixture
def running_data(data_generator):
    """Generate running pattern data."""
    return data_generator.generate_running(
        n_samples=30000,
        step_freq=3.5,
        intensity=0.8
    )


@pytest.fixture
def sleep_data(data_generator):
    """Generate sleep pattern data."""
    return data_generator.generate_sleep(
        n_samples=30000,
        micro_movements=True
    )


@pytest.fixture
def realistic_day_data(data_generator):
    """Generate a full day of realistic data."""
    return data_generator.generate_realistic_day(
        duration_hours=1.0,  # 1 hour for faster tests
        sample_rate=100
    )


@pytest.fixture
def test_dataset_100k():
    """Generate standard 100k sample test dataset."""
    data, metadata = generate_test_dataset(
        n_samples=100000,
        seed=42,
        include_calibration_error=True
    )
    return data, metadata


@pytest.fixture
def test_dataset_no_calib_error():
    """Generate test dataset without calibration error."""
    data, metadata = generate_test_dataset(
        n_samples=100000,
        seed=42,
        include_calibration_error=False
    )
    return data, metadata


@pytest.fixture
def epoch_data_with_features(realistic_day_data, data_generator):
    """Generate epoch-level data with features."""
    return data_generator.generate_epoch_features(
        realistic_day_data,
        epoch_period=30,
        sample_rate=100
    )


@pytest.fixture
def mock_calibration_points(data_generator):
    """Generate multiple stationary periods at different orientations for calibration."""
    # Generate points across the unit sphere to ensure good distribution
    # Need points outside cube with side 0.6 (CALIB_CUBE = 0.3 per axis)
    orientations = [
        (0.0, -1.0, 0.0),      # face down
        (0.0, 1.0, 0.0),       # face up
        (1.0, 0.0, 0.0),       # right side
        (-1.0, 0.0, 0.0),      # left side
        (0.0, 0.0, 1.0),       # top
        (0.0, 0.0, -1.0),      # bottom
        (0.577, -0.577, 0.577),   # diagonal corners
        (-0.577, 0.577, -0.577),
        (0.707, -0.707, 0.0),  # diagonal faces
        (-0.707, 0.707, 0.0),
        (0.707, 0.0, -0.707),
        (0.0, 0.707, 0.707),
    ]

    all_data = []
    for i, orientation in enumerate(orientations):
        data = data_generator.generate_stationary(
            n_samples=200,  # Fewer samples per orientation
            orientation=orientation,
            noise_std=0.008,
            temperature=20.0 + (i % 5) * 2,  # Vary temperature across orientations
            temp_drift=0.001  # Small drift within each sample to provide temperature variation
        )
        all_data.append(data)

    full_data = pd.concat(all_data, ignore_index=True)
    return full_data


@pytest.fixture
def expected_summary_ranges():
    """Define expected ranges for summary metrics (for validation)."""
    return {
        'calibration-errsAfter(mg)': (0.0, 15.0),  # Should be < 10mg ideally
        'wearTime-overall(days)': (0.9, 1.1),  # For 1-day test
        'acc-overall-avg': (15.0, 40.0),  # Typical daily average ENMO
        'sleep-overall-avg': (0.0, 0.5),  # Fraction of time
        'sedentary-overall-avg': (0.2, 0.7),
        'light-overall-avg': (0.0, 0.3),
        'moderate-overall-avg': (0.0, 0.2),
        'vigorous-overall-avg': (0.0, 0.1),
    }


@pytest.fixture
def tolerance_levels():
    """Define numerical tolerance levels for regression tests."""
    return {
        'calibration_params': 1e-4,  # Calibration coefficients
        'summary_metrics': 1e-2,  # Summary statistics (relative)
        'epoch_values': 1e-3,  # Epoch-level values
        'aggregated_stats': 1e-2,  # Daily/hourly aggregations
    }


@pytest.fixture
def temp_output_dir(test_data_dir):
    """Create a temporary output directory for test files."""
    output_dir = test_data_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


# Markers for different test categories
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "regression: marks tests as numerical regression tests"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "requires_java: marks tests that require Java runtime"
    )


@pytest.fixture
def assert_arrays_close():
    """Helper function to assert arrays are numerically close."""
    def _assert_close(actual, expected, rtol=1e-5, atol=1e-8, name="array"):
        """Assert two arrays are close within tolerance."""
        np.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=atol,
            err_msg=f"{name} values differ from expected"
        )
    return _assert_close


@pytest.fixture
def assert_dataframes_close():
    """Helper function to assert DataFrames are numerically close."""
    def _assert_close(actual, expected, rtol=1e-5, atol=1e-8):
        """Assert two DataFrames are close within tolerance."""
        # Check columns match
        assert set(actual.columns) == set(expected.columns), \
            f"Columns differ: {set(actual.columns)} vs {set(expected.columns)}"

        # Check indices match
        pd.testing.assert_index_equal(actual.index, expected.index)

        # Check values for numeric columns
        for col in actual.columns:
            if pd.api.types.is_numeric_dtype(actual[col]):
                np.testing.assert_allclose(
                    actual[col].values,
                    expected[col].values,
                    rtol=rtol, atol=atol,
                    err_msg=f"Column '{col}' differs"
                )
            else:
                pd.testing.assert_series_equal(actual[col], expected[col])

    return _assert_close
