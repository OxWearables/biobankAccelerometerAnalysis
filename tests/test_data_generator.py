"""
Realistic mock data generator for accelerometer testing.

Generates synthetic accelerometer data with realistic patterns for:
- Sleep (low activity, consistent orientation)
- Sedentary (minimal movement, varying orientation)
- Light activity (walking patterns)
- Moderate activity (brisk walking, jogging)
- Vigorous activity (running, high intensity)
- Stationary periods (for calibration testing)
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Dict


class AccelerometerDataGenerator:
    """Generate realistic accelerometer data for testing."""

    def __init__(self, seed: int = 42):
        """
        Initialize data generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.g = 1.0  # Gravitational constant in g units

    def generate_stationary(
        self,
        n_samples: int,
        orientation: Tuple[float, float, float] = (0.0, -1.0, 0.0),
        noise_std: float = 0.005,  # 5mg noise
        temperature: float = 20.0,
        temp_drift: float = 0.0001  # Typical: 0.1°C over 1000 samples
    ) -> pd.DataFrame:
        """
        Generate stationary data (device not moving).

        Args:
            n_samples: Number of samples to generate
            orientation: Base orientation (x, y, z) in g units, should have norm ~1.0
            noise_std: Standard deviation of noise in g units
            temperature: Base temperature in Celsius
            temp_drift: Standard deviation of temperature steps per sample (°C).
                       Forms a random walk where total drift scales as sqrt(n_samples).
                       Example: 0.0001 gives ~0.1°C drift over 1000 samples at 100Hz

        Returns:
            DataFrame with columns: x, y, z, temp
        """
        # Normalize orientation to unit vector
        orientation = np.array(orientation)
        orientation = orientation / np.linalg.norm(orientation)

        # Add small random noise
        x = orientation[0] + self.rng.normal(0, noise_std, n_samples)
        y = orientation[1] + self.rng.normal(0, noise_std, n_samples)
        z = orientation[2] + self.rng.normal(0, noise_std, n_samples)

        # Temperature with slow drift (random walk)
        # temp_drift is the standard deviation of temperature steps per sample
        # Total drift will scale as sqrt(n_samples) for a random walk
        temp = temperature + np.cumsum(self.rng.normal(0, temp_drift, n_samples))

        return pd.DataFrame({'x': x, 'y': y, 'z': z, 'temp': temp})

    def generate_sleep(
        self,
        n_samples: int,
        base_orientation: Tuple[float, float, float] = (-0.3, -0.9, 0.3),
        noise_std: float = 0.008,  # 8mg noise
        micro_movements: bool = True,
        temperature: float = 22.0
    ) -> pd.DataFrame:
        """
        Generate sleep data (minimal movement, occasional position changes).

        Args:
            n_samples: Number of samples
            base_orientation: Base sleep orientation
            noise_std: Noise level
            micro_movements: Include micro-movements (restlessness)
            temperature: Base temperature

        Returns:
            DataFrame with x, y, z, temp
        """
        data = self.generate_stationary(n_samples, base_orientation, noise_std, temperature)

        if micro_movements:
            # Add occasional small movements (position shifts)
            n_movements = int(n_samples / 10000)  # ~1 movement per 100 seconds at 100Hz
            for _ in range(n_movements):
                start = self.rng.randint(0, n_samples - 500)
                duration = self.rng.randint(100, 500)

                # Small orientation shift (same shift applied to all samples in range)
                shift = self.rng.normal(0, 0.05, 3)
                # Note: Broadcasting (3,) array to (n_rows, 3) DataFrame slice
                data.loc[start:start + duration, ['x', 'y', 'z']] += shift

        return data

    def generate_walking(
        self,
        n_samples: int,
        step_freq: float = 2.0,  # steps per second
        intensity: float = 0.3,  # amplitude of movement
        temperature: float = 23.0
    ) -> pd.DataFrame:
        """
        Generate walking pattern with periodic arm swing.

        Args:
            n_samples: Number of samples
            step_freq: Step frequency in Hz
            intensity: Movement amplitude multiplier
            temperature: Base temperature

        Returns:
            DataFrame with x, y, z, temp
        """
        t = np.arange(n_samples) / 100.0  # Time in seconds at 100Hz

        # Base orientation (arm hanging down)
        base_y = -0.8
        base_z = 0.6

        # Periodic arm swing (sinusoidal)
        x = intensity * np.sin(2 * np.pi * step_freq * t)
        y = base_y + intensity * 0.3 * np.cos(2 * np.pi * step_freq * t)
        z = base_z + intensity * 0.2 * np.sin(2 * np.pi * step_freq * 0.5 * t)

        # Add noise
        x += self.rng.normal(0, 0.02, n_samples)
        y += self.rng.normal(0, 0.02, n_samples)
        z += self.rng.normal(0, 0.02, n_samples)

        # Temperature increases slightly with activity
        temp = temperature + np.cumsum(self.rng.normal(0.0001, 0.001, n_samples))

        return pd.DataFrame({'x': x, 'y': y, 'z': z, 'temp': temp})

    def generate_running(
        self,
        n_samples: int,
        step_freq: float = 3.5,  # higher cadence
        intensity: float = 0.8,
        temperature: float = 24.0
    ) -> pd.DataFrame:
        """
        Generate running pattern (higher intensity than walking).

        Args:
            n_samples: Number of samples
            step_freq: Step frequency in Hz
            intensity: Movement amplitude
            temperature: Base temperature

        Returns:
            DataFrame with x, y, z, temp
        """
        t = np.arange(n_samples) / 100.0

        # More vigorous arm movement
        x = intensity * np.sin(2 * np.pi * step_freq * t)
        y = -0.5 + intensity * 0.5 * np.cos(2 * np.pi * step_freq * t)
        z = 0.5 + intensity * 0.4 * np.sin(2 * np.pi * step_freq * 0.5 * t)

        # Higher frequency components (impact)
        x += 0.2 * intensity * np.sin(2 * np.pi * step_freq * 2 * t)

        # More noise from impacts
        x += self.rng.normal(0, 0.05, n_samples)
        y += self.rng.normal(0, 0.05, n_samples)
        z += self.rng.normal(0, 0.05, n_samples)

        # Temperature increases more with vigorous activity
        temp = temperature + np.cumsum(self.rng.normal(0.0005, 0.002, n_samples))

        return pd.DataFrame({'x': x, 'y': y, 'z': z, 'temp': temp})

    def generate_sedentary(
        self,
        n_samples: int,
        activity_type: str = 'sitting',
        temperature: float = 21.0
    ) -> pd.DataFrame:
        """
        Generate sedentary activity (sitting, desk work).

        Args:
            n_samples: Number of samples
            activity_type: Type of sedentary activity
            temperature: Base temperature

        Returns:
            DataFrame with x, y, z, temp
        """
        # Base orientation for wrist on desk/lap
        if activity_type == 'sitting':
            orientation = (0.8, -0.5, 0.3)
        else:
            orientation = (0.5, -0.7, 0.5)

        data = self.generate_stationary(n_samples, orientation, noise_std=0.012, temperature=temperature)

        # Add occasional small movements (typing, mouse clicks)
        n_movements = int(n_samples / 1000)  # ~1 per 10 seconds
        for _ in range(n_movements):
            start = self.rng.randint(0, n_samples - 50)
            duration = self.rng.randint(10, 50)

            # Small quick movements - calculate actual slice length explicitly
            slice_idx = slice(start, start + duration + 1)  # loc is inclusive on both ends
            slice_length = len(data.loc[slice_idx, 'x'])
            data.loc[slice_idx, 'x'] += self.rng.normal(0, 0.03, slice_length)

        return data

    def generate_realistic_day(
        self,
        duration_hours: float = 24.0,
        sample_rate: int = 100,
        start_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Generate a full day of realistic accelerometer data with mixed activities.

        Args:
            duration_hours: Duration in hours
            sample_rate: Sampling rate in Hz
            start_time: Start datetime (default: 2020-06-14 00:00:00)

        Returns:
            DataFrame with time index and x, y, z, temp columns
        """
        if start_time is None:
            start_time = datetime(2020, 6, 14, 0, 0, 0)

        n_samples = int(duration_hours * 3600 * sample_rate)

        # Create time index
        time_index = pd.date_range(
            start=start_time,
            periods=n_samples,
            freq=f'{1000000 // sample_rate}us'  # microseconds
        )

        # Initialize arrays
        all_data = []

        # Typical day schedule (approximate sample counts at 100Hz)
        activities = [
            ('sleep', int(7 * 3600 * sample_rate)),  # 7 hours sleep
            ('sedentary', int(1 * 3600 * sample_rate)),  # 1 hour morning routine
            ('walking', int(0.25 * 3600 * sample_rate)),  # 15 min commute
            ('sedentary', int(4 * 3600 * sample_rate)),  # 4 hours work
            ('walking', int(0.17 * 3600 * sample_rate)),  # 10 min lunch walk
            ('sedentary', int(4 * 3600 * sample_rate)),  # 4 hours work
            ('walking', int(0.25 * 3600 * sample_rate)),  # 15 min commute
            ('sedentary', int(2 * 3600 * sample_rate)),  # 2 hours evening
            ('running', int(0.5 * 3600 * sample_rate)),  # 30 min exercise
            ('sedentary', int(2 * 3600 * sample_rate)),  # 2 hours evening
            ('stationary', int(0.083 * 3600 * sample_rate)),  # 5 min stationary
            ('sleep', int(2.75 * 3600 * sample_rate)),  # Rest of night
        ]

        current_temp = 20.0
        for activity, n_samp in activities:
            if activity == 'sleep':
                data = self.generate_sleep(n_samp, temperature=current_temp)
            elif activity == 'walking':
                data = self.generate_walking(n_samp, intensity=0.3, temperature=current_temp)
            elif activity == 'running':
                data = self.generate_running(n_samp, intensity=0.8, temperature=current_temp)
            elif activity == 'sedentary':
                data = self.generate_sedentary(n_samp, temperature=current_temp)
            elif activity == 'stationary':
                data = self.generate_stationary(n_samp, temperature=current_temp)

            all_data.append(data)
            current_temp = data['temp'].iloc[-1]  # Continue temperature

        # Concatenate all activities
        full_data = pd.concat(all_data, ignore_index=True)

        # Ensure we have exactly the right number of samples
        if len(full_data) > n_samples:
            full_data = full_data.iloc[:n_samples]
        elif len(full_data) < n_samples:
            # Pad with last activity
            padding = n_samples - len(full_data)
            pad_data = self.generate_stationary(padding, temperature=current_temp)
            full_data = pd.concat([full_data, pad_data], ignore_index=True)

        full_data.index = time_index
        full_data.index.name = 'time'

        return full_data

    def add_calibration_error(
        self,
        data: pd.DataFrame,
        offset: Tuple[float, float, float] = (0.01, -0.02, 0.015),
        slope: Tuple[float, float, float] = (0.98, 1.02, 0.99),
        temp_coeff: Tuple[float, float, float] = (0.0001, -0.0002, 0.00015)
    ) -> pd.DataFrame:
        """
        Add calibration error to data (simulating factory calibration drift).

        This function implements the INVERSE of the calibration formula used in
        device.py (lines 304-305). The formula MUST match exactly or all
        calibration tests become invalid.

        Calibration formula (forward, in device.py):
            calibrated = offset + (raw * slope) + (temp * temp_coeff)

        Inverse formula (this function):
            raw = (calibrated - offset - temp*temp_coeff) / slope

        Args:
            data: Input calibrated data
            offset: Offset error per axis (g)
            slope: Slope error per axis (multiplicative)
            temp_coeff: Temperature coefficient per axis (g/°C)

        Returns:
            Uncalibrated data (simulating raw sensor output with calibration drift)

        Note:
            The formula is verified in test_regression.py::test_calibration_error_application
        """
        data = data.copy()

        # Apply inverse calibration (simulate uncalibrated sensor)
        for i, axis in enumerate(['x', 'y', 'z']):
            data[axis] = (data[axis] - offset[i] - data['temp'] * temp_coeff[i]) / slope[i]

        return data

    def generate_epoch_features(
        self,
        raw_data: pd.DataFrame,
        epoch_period: int = 30,
        sample_rate: int = 100
    ) -> pd.DataFrame:
        """
        Generate basic epoch-level features from raw data.

        Args:
            raw_data: Raw accelerometer data
            epoch_period: Epoch duration in seconds
            sample_rate: Sampling rate in Hz

        Returns:
            DataFrame with epoch-level features
        """
        n_samples_per_epoch = epoch_period * sample_rate

        # Resample to epochs
        epoch_data = raw_data.resample(f'{epoch_period}s').agg({
            'x': ['mean', 'std'],
            'y': ['mean', 'std'],
            'z': ['mean', 'std'],
            'temp': 'mean'
        })

        # Flatten column names
        epoch_data.columns = ['_'.join(col).strip() for col in epoch_data.columns.values]

        # Calculate ENMO (Euclidean Norm Minus One, truncated)
        vm = np.sqrt(
            epoch_data['x_mean']**2 +
            epoch_data['y_mean']**2 +
            epoch_data['z_mean']**2
        )
        enmo = vm - 1.0
        epoch_data['enmoTrunc'] = np.maximum(enmo, 0)

        return epoch_data


def generate_test_dataset(
    n_samples: int = 100000,
    seed: int = 42,
    include_calibration_error: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate a complete test dataset with known properties.

    Args:
        n_samples: Number of samples (default 100k = 1000 seconds at 100Hz)
        seed: Random seed
        include_calibration_error: Whether to add calibration error

    Returns:
        Tuple of (raw_data, metadata)
    """
    generator = AccelerometerDataGenerator(seed=seed)

    # Generate mixed activities for 1000 seconds
    duration_hours = n_samples / (100 * 3600)
    start_time = datetime(2020, 6, 14, 10, 0, 0)

    # Create realistic data
    data = generator.generate_realistic_day(
        duration_hours=duration_hours,
        sample_rate=100,
        start_time=start_time
    )

    # Store ground truth
    metadata = {
        'n_samples': n_samples,
        'sample_rate': 100,
        'duration_seconds': n_samples / 100,
        'start_time': start_time,
        'seed': seed,
        'calibrated': not include_calibration_error
    }

    if include_calibration_error:
        # Add known calibration error
        calibration_params = {
            'offset': (0.012, -0.018, 0.015),
            'slope': (0.985, 1.022, 0.994),
            'temp_coeff': (0.0001, -0.00015, 0.00012)
        }
        data = generator.add_calibration_error(data, **calibration_params)
        metadata['calibration_params'] = calibration_params

    return data, metadata


if __name__ == '__main__':
    # Demo: Generate test data
    data, metadata = generate_test_dataset(n_samples=100000)
    print(f"Generated {len(data)} samples")
    print(f"Duration: {metadata['duration_seconds']} seconds")
    print(f"Shape: {data.shape}")
    print("\nFirst few rows:")
    print(data.head())
    print("\nBasic statistics:")
    print(data.describe())

    # Calculate vector magnitude
    vm = np.sqrt(data['x']**2 + data['y']**2 + data['z']**2)
    print(f"\nVector magnitude: mean={vm.mean():.4f}, std={vm.std():.4f}")
    print("Expected ~1.0g for stationary, deviations indicate calibration error")
