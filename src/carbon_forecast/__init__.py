"""carbon_forecast — consumption-based carbon intensity forecasting framework."""

import os

# Pin the Keras 3 backend to TensorFlow (locked decision, CarbonCast lineage)
# before any submodule imports keras, so runs are deterministic regardless of
# environment defaults.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

__version__ = "0.1.0"
