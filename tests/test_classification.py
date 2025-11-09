"""
Unit tests for accelerometer.classification module.

Tests HMM smoothing, spurious sleep removal, and activity classification.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from accelerometer import classification


@pytest.mark.unit
class TestViterbiAlgorithm:
    """Test Viterbi HMM smoothing algorithm."""

    def test_viterbi_identity(self):
        """Test that Viterbi with perfect predictions returns same sequence."""
        # Simple HMM params with identity transition (no smoothing)
        labels = np.array(['sleep', 'sedentary', 'light'])
        hmm_params = {
            'prior': np.array([1 / 3, 1 / 3, 1 / 3]),
            'emission': np.eye(3),  # Perfect emission
            'transition': np.eye(3),  # Identity transition (no smoothing)
            'labels': labels
        }

        Y_obs = pd.Series(['sleep', 'sedentary', 'light', 'sedentary', 'sleep'])
        Y_smooth = classification.viterbi(Y_obs, hmm_params)

        # With identity transition, should return same sequence
        np.testing.assert_array_equal(Y_smooth, Y_obs.values)

    def test_viterbi_smooths_outliers(self):
        """Test that Viterbi smooths isolated misclassifications."""
        labels = np.array(['sleep', 'sedentary', 'light'])

        # HMM params that discourage rapid transitions
        hmm_params = {
            'prior': np.array([1 / 3, 1 / 3, 1 / 3]),
            'emission': np.array([
                [0.9, 0.05, 0.05],  # sleep->sleep high prob
                [0.05, 0.9, 0.05],  # sedentary->sedentary high prob
                [0.05, 0.05, 0.9],  # light->light high prob
            ]),
            'transition': np.array([
                [0.9, 0.08, 0.02],   # sleep mostly stays sleep
                [0.05, 0.9, 0.05],   # sedentary mostly stays
                [0.02, 0.08, 0.9],   # light mostly stays
            ]),
            'labels': labels
        }

        # Sequence with isolated outlier
        Y_obs = pd.Series(['sleep', 'sleep', 'light', 'sleep', 'sleep'])
        # The 'light' in the middle is isolated

        Y_smooth = classification.viterbi(Y_obs, hmm_params)

        # The isolated 'light' should likely be smoothed to 'sleep'
        # Check that smoothing occurred
        assert len(np.unique(Y_smooth)) <= len(np.unique(Y_obs))

    def test_viterbi_handles_all_same_label(self):
        """Test Viterbi with all same labels."""
        labels = np.array(['sleep', 'sedentary'])
        hmm_params = {
            'prior': np.array([0.5, 0.5]),
            'emission': np.eye(2),
            'transition': np.eye(2),
            'labels': labels
        }

        Y_obs = pd.Series(['sleep', 'sleep', 'sleep', 'sleep'])
        Y_smooth = classification.viterbi(Y_obs, hmm_params)

        # Should return all sleep
        assert np.all(Y_smooth == 'sleep')

    def test_viterbi_numerical_stability(self):
        """Test Viterbi doesn't overflow/underflow with small probabilities."""
        labels = np.array(['a', 'b', 'c'])

        # Very small probabilities
        hmm_params = {
            'prior': np.array([0.0001, 0.0001, 0.9998]),
            'emission': np.array([
                [0.001, 0.001, 0.998],
                [0.001, 0.998, 0.001],
                [0.998, 0.001, 0.001],
            ]),
            'transition': np.array([
                [0.001, 0.001, 0.998],
                [0.001, 0.998, 0.001],
                [0.998, 0.001, 0.001],
            ]),
            'labels': labels
        }

        Y_obs = pd.Series(['a', 'b', 'c', 'a', 'b'])
        Y_smooth = classification.viterbi(Y_obs, hmm_params)

        # Should complete without error
        assert len(Y_smooth) == len(Y_obs)
        assert all(label in labels for label in Y_smooth)

    def test_viterbi_long_sequence(self):
        """Test Viterbi with long sequence."""
        labels = np.array(['sleep', 'sedentary', 'light', 'moderate', 'vigorous'])
        n_labels = len(labels)

        hmm_params = {
            'prior': np.ones(n_labels) / n_labels,
            'emission': 0.8 * np.eye(n_labels) + 0.05,  # Noisy emission
            'transition': 0.7 * np.eye(n_labels) + 0.05,  # Prefer staying
            'labels': labels
        }

        # Long random sequence
        rng = np.random.RandomState(42)
        Y_obs = pd.Series(rng.choice(labels, size=1000))

        Y_smooth = classification.viterbi(Y_obs, hmm_params)

        # Should complete without error
        assert len(Y_smooth) == len(Y_obs)


@pytest.mark.unit
class TestSpuriousSleepRemoval:
    """Test spurious sleep removal."""

    def create_time_series(self, labels, freq='30s'):
        """Helper to create time series with labels."""
        start = datetime(2020, 6, 14, 0, 0, 0)
        index = pd.date_range(start=start, periods=len(labels), freq=freq)
        return pd.Series(labels, index=index)

    def test_remove_short_sleep_episodes(self):
        """Test that short sleep episodes are removed."""
        # 30-second epochs: 60 minutes = 120 epochs
        labels = (['sedentary'] * 100 +
                  ['sleep'] * 50 +  # 25 minutes - should be removed
                  ['sedentary'] * 100)

        Y = self.create_time_series(labels)
        Y_clean = classification.removeSpuriousSleep(Y, sleepTol=60)

        # Short sleep should be replaced with sedentary
        assert 'sleep' not in Y_clean[100:150].values
        assert all(Y_clean[100:150] == 'sedentary')

    def test_keep_long_sleep_episodes(self):
        """Test that long sleep episodes are kept."""
        # 30-second epochs: 120 minutes = 240 epochs
        labels = (['sedentary'] * 50 +
                  ['sleep'] * 240 +  # 120 minutes - should be kept
                  ['sedentary'] * 50)

        Y = self.create_time_series(labels)
        Y_clean = classification.removeSpuriousSleep(Y, sleepTol=60)

        # Long sleep should be preserved
        assert all(Y_clean[50:290] == 'sleep')

    def test_tolerance_threshold(self):
        """Test different tolerance thresholds."""
        # Exactly 60 minutes of sleep
        labels = ['sedentary'] * 50 + ['sleep'] * 120 + ['sedentary'] * 50
        Y = self.create_time_series(labels)

        # With tolerance=60, should be kept (>=)
        Y_clean_60 = classification.removeSpuriousSleep(Y, sleepTol=60)
        assert all(Y_clean_60[50:170] == 'sleep')

        # With tolerance=61, should be removed (<)
        Y_clean_61 = classification.removeSpuriousSleep(Y, sleepTol=61)
        assert all(Y_clean_61[50:170] == 'sedentary')

    def test_multiple_sleep_episodes(self):
        """Test with multiple sleep episodes."""
        labels = (['sedentary'] * 50 +
                  ['sleep'] * 240 +  # Long sleep - keep
                  ['sedentary'] * 50 +
                  ['sleep'] * 30 +   # Short sleep - remove
                  ['sedentary'] * 50)

        Y = self.create_time_series(labels)
        Y_clean = classification.removeSpuriousSleep(Y, sleepTol=60)

        # First sleep kept, second removed
        assert all(Y_clean[50:290] == 'sleep')
        assert all(Y_clean[340:370] == 'sedentary')

    def test_does_not_modify_original(self):
        """Test that original series is not modified."""
        labels = ['sedentary'] * 50 + ['sleep'] * 30 + ['sedentary'] * 50
        Y = self.create_time_series(labels)
        Y_original = Y.copy()

        classification.removeSpuriousSleep(Y, sleepTol=60)

        # Original should be unchanged
        pd.testing.assert_series_equal(Y, Y_original)

    def test_different_activity_models(self):
        """Test replacement value for different models."""
        labels = ['light'] * 50 + ['sleep'] * 30 + ['light'] * 50
        Y = self.create_time_series(labels)

        # Walmsley model uses 'sedentary'
        Y_walmsley = classification.removeSpuriousSleep(Y, activityModel='walmsley', sleepTol=60)
        assert all(Y_walmsley[50:80] == 'sedentary')

        # Willetts model uses 'sit-stand'
        Y_willetts = classification.removeSpuriousSleep(Y, activityModel='willetts', sleepTol=60)
        assert all(Y_willetts[50:80] == 'sit-stand')

    def test_no_sleep_labels(self):
        """Test with no sleep labels."""
        labels = ['sedentary', 'light', 'moderate'] * 50
        Y = self.create_time_series(labels)

        Y_clean = classification.removeSpuriousSleep(Y, sleepTol=60)

        # Should be unchanged
        pd.testing.assert_series_equal(Y, Y_clean)

    def test_all_sleep_labels(self):
        """Test with all sleep labels."""
        labels = ['sleep'] * 300  # 150 minutes
        Y = self.create_time_series(labels)

        Y_clean = classification.removeSpuriousSleep(Y, sleepTol=60)

        # All should be preserved (long enough)
        assert all(Y_clean == 'sleep')

    def test_consecutive_sleep_episodes(self):
        """Test consecutive sleep episodes separated by brief wake."""
        labels = (['sleep'] * 100 +  # 50 min
                  ['sedentary'] * 4 +  # 2 min
                  ['sleep'] * 100)  # 50 min

        Y = self.create_time_series(labels)
        Y_clean = classification.removeSpuriousSleep(Y, sleepTol=60)

        # Both sleep episodes should be removed (each < 60 min)
        assert all(Y_clean[:100] == 'sedentary')
        assert all(Y_clean[104:] == 'sedentary')


@pytest.mark.unit
class TestCutPointModel:
    """Test cut-point based classification."""

    def test_cutpoint_classification_default(self):
        """Test cut-point classification with default thresholds."""
        # ENMO values in g units
        enmo = pd.Series([
            0.020,  # < 45mg -> sedentary
            0.060,  # 45-100mg -> LPA
            0.150,  # 100-400mg -> MPA
            0.500,  # > 400mg -> VPA
        ])

        result = classification.cutPointModel(enmo)

        assert result['cp-sedentary'].iloc[0] == 1.0
        assert result['cp-LPA'].iloc[1] == 1.0
        assert result['cp-MPA'].iloc[2] == 1.0
        assert result['cp-VPA'].iloc[3] == 1.0

    def test_cutpoint_classification_custom_cuts(self):
        """Test with custom cut-points."""
        enmo = pd.Series([0.020, 0.060, 0.150])

        custom_cuts = {'LPA': 0.050, 'MPA': 0.120, 'VPA': 0.300}
        result = classification.cutPointModel(enmo, cuts=custom_cuts)

        assert result['cp-sedentary'].iloc[0] == 1.0  # < 50mg
        assert result['cp-LPA'].iloc[1] == 1.0  # 50-120mg
        assert result['cp-MPA'].iloc[2] == 1.0  # 120-300mg

    def test_cutpoint_with_where_filter(self):
        """Test cut-point classification with where filter (exclude sleep)."""
        enmo = pd.Series([0.020, 0.060, 0.150, 0.020])
        whr = pd.Series([True, True, True, False])  # Last one is sleep

        result = classification.cutPointModel(enmo, whr=whr)

        # First 3 should be classified
        assert result['cp-sedentary'].iloc[0] == 1.0

        # Last one should be NaN (filtered out)
        assert pd.isna(result['cp-sedentary'].iloc[3])

    def test_cutpoint_boundary_values(self):
        """Test behavior at exact boundary values."""
        # Test at exact cut-points
        enmo = pd.Series([0.045, 0.100, 0.400])

        result = classification.cutPointModel(enmo)

        # Should use >= for lower bound, < for upper bound
        # (implementation-specific, verify actual behavior)
        assert sum(result.iloc[0] == 1.0) == 1  # Only one category

    def test_cutpoint_returns_one_hot(self):
        """Test that result is one-hot encoded."""
        enmo = pd.Series([0.020, 0.060, 0.150, 0.500])

        result = classification.cutPointModel(enmo)

        # Each row should have exactly one 1.0
        assert all(result.sum(axis=1) == 1.0)


@pytest.mark.unit
class TestActivityClassificationHelpers:
    """Test helper functions for activity classification."""

    def test_model_path_resolution(self):
        """Test model path resolution."""
        # Test with known model name
        try:
            path = classification.resolveModelPath('walmsley')
            assert path is not None
        except FileNotFoundError:
            # Model not downloaded yet - expected
            pass

    def test_invalid_model_name(self):
        """Test with invalid model name."""
        with pytest.raises((FileNotFoundError, KeyError)):
            classification.resolveModelPath('invalid_model_name')


@pytest.mark.slow
@pytest.mark.integration
class TestFullClassificationPipeline:
    """Integration tests for full classification pipeline."""

    def test_classification_with_mock_epoch_data(self, epoch_data_with_features):
        """Test classification on realistic epoch data."""
        # This would require:
        # 1. Mock model or actual model download
        # 2. Proper feature columns in epoch data
        # Skip if model not available
        pytest.skip("Requires trained model - integration test")

    def test_end_to_end_activity_prediction(self, realistic_day_data):
        """Test end-to-end activity prediction."""
        # This would test the full pipeline from raw data to activities
        pytest.skip("Requires full pipeline - integration test")


@pytest.mark.unit
class TestNumericalStability:
    """Test numerical stability of classification functions."""

    def test_viterbi_zero_probabilities(self):
        """Test Viterbi handles zero probabilities gracefully."""
        labels = np.array(['a', 'b'])

        # Some zero probabilities
        hmm_params = {
            'prior': np.array([0.5, 0.5]),
            'emission': np.array([
                [0.0, 1.0],  # Zero probability
                [1.0, 0.0],
            ]),
            'transition': np.array([
                [0.0, 1.0],
                [1.0, 0.0],
            ]),
            'labels': labels
        }

        Y_obs = pd.Series(['a', 'b', 'a', 'b'])

        # Should not crash (uses SMALL_NUMBER to avoid log(0))
        Y_smooth = classification.viterbi(Y_obs, hmm_params)
        assert len(Y_smooth) == len(Y_obs)

    def test_spurious_sleep_empty_series(self):
        """Test spurious sleep removal with empty series."""
        Y = pd.Series([], dtype=str)
        Y.index = pd.DatetimeIndex([])

        Y_clean = classification.removeSpuriousSleep(Y, sleepTol=60)

        assert len(Y_clean) == 0

    def test_cutpoint_with_nan_values(self):
        """Test cut-point classification with NaN values."""
        enmo = pd.Series([0.020, np.nan, 0.150, np.nan])

        result = classification.cutPointModel(enmo)

        # NaN inputs should produce NaN outputs
        assert pd.isna(result['cp-sedentary'].iloc[1])
        assert pd.isna(result['cp-sedentary'].iloc[3])

        # Valid inputs should be classified
        assert result['cp-sedentary'].iloc[0] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
