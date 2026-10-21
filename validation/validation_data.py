import os
import io
from PIL import Image, UnidentifiedImageError

class DataValidationLayer:
    """
    Data Validation Layer to reject bad, corrupted, or out-of-spec image data 
    early before it reaches model inference or training pipelines.
    """
    def __init__(self, target_size=(512, 512), max_file_size_mb=10, min_resolution=64):
        self.target_size = target_size
        self.max_file_size_mb = max_file_size_mb
        self.min_resolution = min_resolution
        self.allowed_formats = {"JPEG", "JPG", "PNG"}

    def validate_image_bytes(self, image_bytes: bytes) -> tuple[bool, str]:
        """
        Validates raw image bytes against size, corruption, format, and resolution criteria.
        Returns: (is_valid: bool, error_message: str)
        """
        # 1. Check for empty payload
        if not image_bytes or len(image_bytes) == 0:
            return False, "Received empty image payload."

        # 2. Check maximum file size limit
        file_size_mb = len(image_bytes) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            return False, f"Image size ({file_size_mb:.2f} MB) exceeds maximum allowed limit ({self.max_file_size_mb} MB)."

        # 3. Check for valid image format and corruption via PIL verify()
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.verify() # Checks file integrity headers
        except UnidentifiedImageError:
            return False, "The provided payload is not a valid image file."
        except Exception as e:
            return False, f"Corrupted or invalid image format: {str(e)}"

        # Re-open image since verify() modifies/closes the internal file pointer
        try:
            image = Image.open(io.BytesIO(image_bytes))
            
            # 4. Enforce allowed image formats
            if image.format not in self.allowed_formats:
                return False, f"Unsupported image format '{image.format}'. Allowed formats: {self.allowed_formats}"

            # 5. Check color mode support
            if image.mode not in ["RGB", "L", "RGBA"]:
                return False, f"Unsupported image color mode: {image.mode}"
                
            # 6. Check resolution / aspect ratio constraints
            width, height = image.size
            if width < self.min_resolution or height < self.min_resolution:
                return False, f"Image resolution too low ({width}x{height}). Minimum required is {self.min_resolution}x{self.min_resolution}."
                
        except Exception as e:
            return False, f"Failed to process image structure: {str(e)}"

        return True, "Validation passed successfully."

# Instantiate global validator for modular imports
validator = DataValidationLayer()