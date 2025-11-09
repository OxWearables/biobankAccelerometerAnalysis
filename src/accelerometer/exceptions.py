"""Custom exception classes for accelerometer processing.

This module defines a hierarchy of exceptions used throughout the
accelerometer processing pipeline to handle errors gracefully without
calling sys.exit() in library code.
"""


class AccelerometerException(Exception):
    """Base exception for all accelerometer processing errors.

    All custom exceptions in this package inherit from this base class,
    allowing users to catch all accelerometer-related errors with a single
    except clause if desired.
    """
    pass


class CalibrationError(AccelerometerException):
    """Raised when calibration process fails.

    This can occur when:
    - Insufficient stationary periods are found in the data
    - Calibration optimization fails to converge
    - Calibration coefficients are outside acceptable ranges
    """
    pass


class DeviceError(AccelerometerException):
    """Raised when device-specific operations fail.

    This can occur when:
    - Device ID cannot be extracted from file header
    - File format is not recognized or is corrupted
    - Device-specific parsing fails
    """
    pass


class ProcessingError(AccelerometerException):
    """Raised when data processing fails.

    This can occur when:
    - Java subprocess fails during processing
    - Data filtering or resampling fails
    - Feature extraction fails
    """
    pass


class DataError(AccelerometerException):
    """Raised when there are issues with the data itself.

    This can occur when:
    - No data remains after filtering
    - Data is corrupt or malformed
    - Required data columns are missing
    """
    pass


class ClassificationError(AccelerometerException):
    """Raised when activity classification fails.

    This can occur when:
    - ML model cannot be loaded
    - Required features are missing
    - Model prediction fails
    """
    pass


class SummarisationError(AccelerometerException):
    """Raised when data summarisation fails.

    This can occur when:
    - Insufficient data for summary statistics
    - Aggregation operations fail
    - Output format generation fails
    """
    pass
