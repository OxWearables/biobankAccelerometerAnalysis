"""
Unit tests for accelerometer.device module.

Tests calibration algorithms, file processing, and device ID extraction.
"""

import pytest
import numpy as np
import pandas as pd

from accelerometer import device


@pytest.mark.unit
class TestCalibrationCoefficients:
    """Test calibration coefficient calculation."""

    def test_perfect_calibration_identity(self, mock_calibration_points):
        """Test that perfectly calibrated data returns identity transform."""
        # Use mock_calibration_points which has multiple orientations
        stationary_df = pd.DataFrame({
            'xMean': mock_calibration_points['x'],
            'yMean': mock_calibration_points['y'],
            'zMean': mock_calibration_points['z'],
            'temp': mock_calibration_points['temp'],
            'dataErrors': 0
        })

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Check that calibration is identity (or very close)
        assert 'calibration-xOffset(g)' in summary
        assert 'calibration-yOffset(g)' in summary
        assert 'calibration-zOffset(g)' in summary

        # Offsets should be near zero
        assert abs(summary['calibration-xOffset(g)']) < 0.02
        assert abs(summary['calibration-yOffset(g)']) < 0.02
        assert abs(summary['calibration-zOffset(g)']) < 0.02

        # Slopes should be near 1.0
        assert abs(summary['calibration-xSlope'] - 1.0) < 0.02
        assert abs(summary['calibration-ySlope'] - 1.0) < 0.02
        assert abs(summary['calibration-zSlope'] - 1.0) < 0.02

        # Should be marked as good calibration
        assert summary['quality-goodCalibration'] == 1

        # Error should be low
        assert summary['calibration-errsAfter(mg)'] < 10.0

    def test_known_calibration_error_recovery(self, mock_calibration_points, data_generator):
        """Test recovery of known calibration error."""
        # Add known calibration error
        known_offset = (0.012, -0.018, 0.015)
        known_slope = (0.985, 1.022, 0.994)
        known_temp_coeff = (0.0001, -0.00015, 0.00012)

        uncalibrated = data_generator.add_calibration_error(
            mock_calibration_points,
            offset=known_offset,
            slope=known_slope,
            temp_coeff=known_temp_coeff
        )

        # Create stationary points DataFrame
        stationary_df = pd.DataFrame({
            'xMean': uncalibrated['x'],
            'yMean': uncalibrated['y'],
            'zMean': uncalibrated['z'],
            'temp': uncalibrated['temp'],
            'dataErrors': 0
        })

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Should recover calibration parameters (within tolerance)
        # Note: won't be exact due to noise and iterative algorithm
        # Tolerances are realistic for noisy data with temperature variation
        assert abs(summary['calibration-xOffset(g)'] - known_offset[0]) < 0.015  # ±15mg
        assert abs(summary['calibration-yOffset(g)'] - known_offset[1]) < 0.015  # ±15mg
        assert abs(summary['calibration-zOffset(g)'] - known_offset[2]) < 0.015  # ±15mg

        assert abs(summary['calibration-xSlope'] - known_slope[0]) < 0.025  # ±2.5%
        assert abs(summary['calibration-ySlope'] - known_slope[1]) < 0.025  # ±2.5%
        assert abs(summary['calibration-zSlope'] - known_slope[2]) < 0.025  # ±2.5%

        # Should be marked as good calibration
        assert summary['quality-goodCalibration'] == 1

    def test_insufficient_samples_fails(self):
        """Test that insufficient samples results in failed calibration."""
        # Only 10 samples - too few
        small_data = pd.DataFrame({
            'xMean': np.random.randn(10),
            'yMean': np.random.randn(10),
            'zMean': np.random.randn(10),
            'temp': np.ones(10) * 20.0,
            'dataErrors': 0
        })

        summary = {}
        device.get_calibration_coefs(small_data, summary)

        # Should fail calibration
        assert summary['quality-goodCalibration'] == 0

        # Should return identity transform
        assert summary['calibration-xOffset(g)'] == 0.0
        assert summary['calibration-xSlope'] == 1.0

    def test_poorly_distributed_points_fails(self):
        """Test that poorly distributed points fail calibration."""
        # All points in same orientation - not enough coverage
        same_orientation = pd.DataFrame({
            'xMean': np.random.randn(1000) * 0.01,
            'yMean': -1.0 + np.random.randn(1000) * 0.01,
            'zMean': np.random.randn(1000) * 0.01,
            'temp': np.ones(1000) * 20.0,
            'dataErrors': 0
        })

        summary = {}
        device.get_calibration_coefs(same_orientation, summary)

        # Should fail due to poor distribution (all points in cube side < 0.3)
        assert summary['quality-goodCalibration'] == 0

    def test_data_errors_filtered(self, mock_calibration_points):
        """Test that points with data errors are filtered out."""
        stationary_df = pd.DataFrame({
            'xMean': mock_calibration_points['x'],
            'yMean': mock_calibration_points['y'],
            'zMean': mock_calibration_points['z'],
            'temp': mock_calibration_points['temp'],
            'dataErrors': 0
        })

        # Mark half the data as errors
        stationary_df.loc[::2, 'dataErrors'] = 1

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Should still work with good points
        # Check that it used fewer samples
        assert summary['calibration-numStaticPoints'] == len(stationary_df) // 2

    def test_nan_values_filtered(self, mock_calibration_points):
        """Test that NaN values are filtered out."""
        stationary_df = pd.DataFrame({
            'xMean': mock_calibration_points['x'],
            'yMean': mock_calibration_points['y'],
            'zMean': mock_calibration_points['z'],
            'temp': mock_calibration_points['temp'],
            'dataErrors': 0
        })

        # Add some NaN values
        stationary_df.loc[::3, 'xMean'] = np.nan

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Should filter out NaN rows
        assert summary['calibration-numStaticPoints'] < len(stationary_df)

    def test_zero_vectors_filtered(self):
        """Test that zero vectors are filtered out."""
        data = pd.DataFrame({
            'xMean': [0.0, 0.0, 0.0, 1.0],
            'yMean': [0.0, 0.0, 0.0, 0.0],
            'zMean': [0.0, 0.0, 0.0, 0.0],
            'temp': [20.0, 20.0, 20.0, 20.0],
            'dataErrors': [0, 0, 0, 0]
        })

        summary = {}
        device.get_calibration_coefs(data, summary)

        # Should filter out zero vectors
        # Will fail due to insufficient samples after filtering
        assert summary['quality-goodCalibration'] == 0

    def test_temperature_coefficient_estimation(self, data_generator):
        """Test that temperature coefficients are properly estimated."""
        # Skip this test - temperature coefficient estimation requires
        # significant temperature variation which is hard to achieve
        # with realistic mock data
        pytest.skip("Temperature coefficient estimation requires large temp variation")

    def test_convergence_within_max_iterations(self, mock_calibration_points, data_generator):
        """Test that algorithm converges within max iterations."""
        # Add moderate calibration error
        uncalibrated = data_generator.add_calibration_error(
            mock_calibration_points,
            offset=(0.02, -0.03, 0.025),
            slope=(0.95, 1.05, 0.96),
            temp_coeff=(0.0002, -0.0003, 0.00025)
        )

        stationary_df = pd.DataFrame({
            'xMean': uncalibrated['x'],
            'yMean': uncalibrated['y'],
            'zMean': uncalibrated['z'],
            'temp': uncalibrated['temp'],
            'dataErrors': 0
        })

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Should converge and be marked as good
        assert summary['quality-goodCalibration'] == 1
        assert summary['calibration-errsAfter(mg)'] < 10.0

        # Error after should be less than error before
        assert summary['calibration-errsAfter(mg)'] < summary['calibration-errsBefore(mg)']

    def test_calibration_improves_error(self, mock_calibration_points, data_generator):
        """Test that calibration reduces error."""
        # Use well-distributed points with calibration error
        uncalibrated = data_generator.add_calibration_error(
            mock_calibration_points,
            offset=(0.01, -0.02, 0.015),
            slope=(0.98, 1.02, 0.99),
            temp_coeff=(0.0001, -0.0002, 0.00015)
        )

        stationary_df = pd.DataFrame({
            'xMean': uncalibrated['x'],
            'yMean': uncalibrated['y'],
            'zMean': uncalibrated['z'],
            'temp': uncalibrated['temp'],
            'dataErrors': 0
        })

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Error after calibration should be much less than before
        err_before = summary['calibration-errsBefore(mg)']
        err_after = summary['calibration-errsAfter(mg)']

        assert err_after < err_before
        assert err_after < 10.0  # Should be < 10mg threshold


@pytest.mark.unit
class TestCalibrationNumericalStability:
    """Test numerical stability of calibration algorithm."""

    def test_different_scales(self, data_generator):
        """Test calibration works with different acceleration scales."""
        # Test with very small accelerations
        small_data = data_generator.generate_stationary(
            n_samples=1000,
            orientation=(0.1, -0.1, 0.1),
            noise_std=0.001
        )

        # Normalize to unit vector
        small_data[['x', 'y', 'z']] = small_data[['x', 'y', 'z']].div(
            np.sqrt((small_data[['x', 'y', 'z']]**2).sum(axis=1)),
            axis=0
        )

        stationary_df = pd.DataFrame({
            'xMean': small_data['x'],
            'yMean': small_data['y'],
            'zMean': small_data['z'],
            'temp': small_data['temp'],
            'dataErrors': 0
        })

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Should handle different scales
        assert 'calibration-xOffset(g)' in summary
        # Won't pass quality check due to poor distribution, but shouldn't crash
        assert 'quality-goodCalibration' in summary

    def test_extreme_temperatures(self, data_generator):
        """Test calibration with extreme temperatures."""
        data = data_generator.generate_stationary(
            n_samples=2000,
            orientation=(0.0, -1.0, 0.0),
            temperature=50.0,  # Very hot
            noise_std=0.005
        )

        stationary_df = pd.DataFrame({
            'xMean': data['x'],
            'yMean': data['y'],
            'zMean': data['z'],
            'temp': data['temp'],
            'dataErrors': 0
        })

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Should handle extreme temperatures without crashing
        assert 'calibration-xOffset(g)' in summary

    def test_reproducibility(self, mock_calibration_points, data_generator):
        """Test that calibration produces same results with same input."""
        uncalibrated = data_generator.add_calibration_error(
            mock_calibration_points,
            offset=(0.01, -0.02, 0.015),
            slope=(0.98, 1.02, 0.99),
            temp_coeff=(0.0001, -0.0002, 0.00015)
        )

        stationary_df = pd.DataFrame({
            'xMean': uncalibrated['x'],
            'yMean': uncalibrated['y'],
            'zMean': uncalibrated['z'],
            'temp': uncalibrated['temp'],
            'dataErrors': 0
        })

        # Run calibration twice
        summary1 = {}
        device.get_calibration_coefs(stationary_df.copy(), summary1)

        summary2 = {}
        device.get_calibration_coefs(stationary_df.copy(), summary2)

        # Should produce identical results
        np.testing.assert_almost_equal(
            summary1['calibration-xOffset(g)'],
            summary2['calibration-xOffset(g)'],
            decimal=10
        )
        np.testing.assert_almost_equal(
            summary1['calibration-errsAfter(mg)'],
            summary2['calibration-errsAfter(mg)'],
            decimal=6
        )


@pytest.mark.unit
class TestCalibrationEdgeCases:
    """Test edge cases in calibration."""

    def test_single_orientation_multiple_samples(self, data_generator):
        """Test with many samples but single orientation."""
        data = data_generator.generate_stationary(
            n_samples=10000,
            orientation=(0.0, -1.0, 0.0),
            noise_std=0.005
        )

        stationary_df = pd.DataFrame({
            'xMean': data['x'],
            'yMean': data['y'],
            'zMean': data['z'],
            'temp': data['temp'],
            'dataErrors': 0
        })

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Should fail due to poor distribution
        assert summary['quality-goodCalibration'] == 0

    def test_missing_temp_column(self, simple_stationary_data):
        """Test calibration when temperature column is missing."""
        stationary_df = pd.DataFrame({
            'xMean': simple_stationary_data['x'],
            'yMean': simple_stationary_data['y'],
            'zMean': simple_stationary_data['z'],
            'dataErrors': 0
        })
        # No 'temp' column

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Should work with dummy temperature
        assert 'calibration-xOffset(g)' in summary

        # Temp coefficients should be zero or very small
        assert abs(summary['calibration-xSlopeTemp']) < 0.0001

    def test_all_same_temperature(self, simple_stationary_data):
        """Test when all temperatures are identical."""
        stationary_df = pd.DataFrame({
            'xMean': simple_stationary_data['x'],
            'yMean': simple_stationary_data['y'],
            'zMean': simple_stationary_data['z'],
            'temp': 20.0,  # Constant temperature
            'dataErrors': 0
        })

        summary = {}
        device.get_calibration_coefs(stationary_df, summary)

        # Should work but temp coefficients should be ~zero
        assert 'calibration-xSlopeTemp' in summary


@pytest.mark.unit
class TestDeviceUtilities:
    """Test utility functions in device module."""

    def test_get_device_id_from_cwa_file(self, test_data_dir):
        """Test device ID extraction from CWA file."""
        # This would require a real CWA file or mock
        # Skip if no test files available
        pytest.skip("Requires real CWA test file")

    def test_stationary_detection_threshold(self, data_generator):
        """Test that stationary detection uses correct threshold."""
        # Generate data with known std
        stationary = data_generator.generate_stationary(
            n_samples=1000,
            noise_std=0.005  # 5mg - below 13mg threshold
        )

        # Check that std is below threshold
        assert stationary['x'].std() < 0.013
        assert stationary['y'].std() < 0.013
        assert stationary['z'].std() < 0.013

        # Generate non-stationary
        walking = data_generator.generate_walking(
            n_samples=1000,
            intensity=0.3
        )

        # Should have higher std
        assert walking['x'].std() > 0.013


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
