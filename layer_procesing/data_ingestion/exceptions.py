# data_ingestion/exceptions.py
class DataIngestionError(Exception):
    """Base exception for data ingestion errors."""
    pass

class LayerNotFoundError(DataIngestionError):
    """Raised when the expected layer is missing in a GeoPackage."""
    def __init__(self, layer_name, filepath):
        message = f"Layer '{layer_name}' not found in GeoPackage: {filepath}"
        super().__init__(message)

class GeoPackageReadError(DataIngestionError):
    """Raised when there is an error reading the GeoPackage file."""
    def __init__(self, filepath, original_exception):
        message = f"Error reading GeoPackage {filepath}: {original_exception}"
        super().__init__(message)