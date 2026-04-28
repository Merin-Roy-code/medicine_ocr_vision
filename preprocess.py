"""
preprocess.py — OpenCV-based image preprocessing pipeline for medicine strips.

Pipeline (in order):
  1. Upscale small images so OCR can read fine print.
  2. Convert BGR → LAB color space.
  3. Apply CLAHE on the L (luminance) channel to enhance local contrast.
  4. Convert back to BGR and apply fast NL-means colour denoising.
  5. Apply a mild unsharp-mask sharpening kernel to sharpen edges.
  6. Normalize lighting with a Gaussian-blur divide to reduce glare / reflections.

Intentionally avoids binarization / heavy thresholding to preserve thin text.
"""

import cv2
import numpy as np
from loguru import logger

import config


def load_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode raw image bytes into an OpenCV BGR ndarray.

    Args:
        image_bytes: Raw bytes from an uploaded image file.

    Returns:
        BGR image as a numpy ndarray.

    Raises:
        ValueError: If the bytes cannot be decoded as an image.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image bytes — ensure the file is a valid image.")
    return img


def upscale_if_small(img: np.ndarray, target_long_edge: int = config.PREPROCESS_TARGET_SIZE) -> np.ndarray:
    """
    Upscale an image so its longest dimension equals *target_long_edge*.
    Images already larger are returned unchanged (we never downscale here).

    Upscaling uses INTER_CUBIC for quality over INTER_LINEAR.
    """
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge >= target_long_edge:
        return img  # already large enough

    scale = target_long_edge / long_edge
    new_w, new_h = int(w * scale), int(h * scale)
    upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    logger.debug(f"Upscaled image from {w}×{h} → {new_w}×{new_h}")
    return upscaled


def apply_clahe(img: np.ndarray, clip_limit: float = config.CLAHE_CLIP_LIMIT) -> np.ndarray:
    """
    Enhance local contrast using CLAHE on the L channel of LAB color space.

    Converting to LAB lets us boost contrast *independently of colour* so that
    foil / metallic backgrounds don't get blown out.

    Args:
        img:        BGR image.
        clip_limit: CLAHE clip limit; higher → stronger contrast (2.0 is mild).

    Returns:
        BGR image with enhanced contrast.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_chan)

    lab_enhanced = cv2.merge([l_enhanced, a_chan, b_chan])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray) -> np.ndarray:
    """
    Remove high-frequency noise using Fast NL-Means Denoising (coloured variant).

    Conservative h=6 preserves fine print while removing speckle noise from
    blister packaging and foil reflections.
    """
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=config.DENOISE_H,
        hColor=config.DENOISE_H,
        templateWindowSize=config.DENOISE_TEMPLATE_WINDOW,
        searchWindowSize=config.DENOISE_SEARCH_WINDOW,
    )


def sharpen(img: np.ndarray) -> np.ndarray:
    """
    Apply a mild unsharp-mask sharpening kernel.

    Kernel is tuned to strengthen character edges without amplifying noise.
    Uses a 5×5 Gaussian blur as the "blurred" reference (gentler than 3×3).
    """
    # Gaussian-blurred version as the low-frequency reference
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Unsharp mask: original + weight × (original - blurred)
    sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharpened


def normalize_lighting(img: np.ndarray) -> np.ndarray:
    """
    Reduce uneven illumination / glare caused by reflective foil surfaces.

    Technique: divide each channel by a large-kernel Gaussian blur of that
    channel.  This is a standard "background estimation" trick — the blurred
    version approximates the low-frequency illumination component, and dividing
    it out leaves only the high-frequency text/detail.
    """
    img_float = img.astype(np.float32)

    # Large kernel (61×61) approximates the global illumination field
    background = cv2.GaussianBlur(img_float, (61, 61), 0)

    # Avoid division-by-zero with a small epsilon
    normalized = (img_float / (background + 1e-6)) * 128
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    return normalized


def preprocess(image_bytes: bytes) -> tuple[np.ndarray, bytes]:
    """
    Full preprocessing pipeline.  Entry point used by ocr.py.

    Args:
        image_bytes: Raw bytes of the uploaded image.

    Returns:
        Tuple of:
          - preprocessed BGR numpy array (for debug / saving)
          - PNG-encoded bytes of the preprocessed image (for Vision API)
    """
    logger.info("Starting image preprocessing pipeline …")

    img = load_image(image_bytes)
    logger.debug(f"Original image shape: {img.shape}")

    img = upscale_if_small(img)
    img = apply_clahe(img)
    img = denoise(img)
    img = sharpen(img)
    img = normalize_lighting(img)

    # Encode to PNG bytes for the Vision API
    success, buffer = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Failed to encode preprocessed image to PNG.")

    preprocessed_bytes = buffer.tobytes()
    logger.info(f"Preprocessing complete — final shape: {img.shape}")
    return img, preprocessed_bytes
