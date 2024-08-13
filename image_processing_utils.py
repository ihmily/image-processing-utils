import base64
import os

from PIL import Image, ImageOps
import cv2
import numpy as np
from typing import Union, Optional


def crop_image_by_alpha_channel(
        input_image: Optional[Union[str, Image.Image, np.ndarray]] = None,
        base64_image: Optional[str] = None,
        output_path: Optional[str] = None,
        file_format: Optional[str] = 'png',
        return_np: Optional[bool] = False,
) -> Union[str, np.ndarray]:
    """
    Crop an image based on its alpha channel.

    Args:
    - input_image: Path to the image, PIL Image object, or NumPy array.
    - base64_image: Optional, base64 encoded image data.
    - output_path: Path to save the cropped image.
    - return_np: (True) If you want the result as a numpy array, return it directly. Default is False.

    Returns:
    - The path where the cropped image is saved, or the numpy array of the cropped image if `return_np` is True.
    """
    if base64_image is not None:
        img_data = base64.b64decode(base64_image)
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        img_array = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)

    elif input_image is not None:
        if isinstance(input_image, str):
            img_array = cv2.imdecode(np.fromfile(input_image, dtype=np.uint8), -1)
        elif isinstance(input_image, Image.Image):
            img_array = np.array(input_image)
        elif isinstance(input_image, np.ndarray):
            img_array = input_image
        else:
            raise ValueError("Invalid input: input_image must be a path, a PIL Image or a numpy array")
    else:
        raise ValueError("Invalid input: input_image or base64_image must be provided")

    if len(img_array.shape) < 3:
        # raise ValueError("Input image does not have an alpha channel")
        print("Warning: Input image does not have an alpha channel")
        cv2.imencode('.png', img_array)[1].tofile(output_path)
        return output_path

    if img_array.shape[2] != 4:
        raise ValueError("Input image must have an alpha channel")

    alpha_channel = img_array[:, :, 3]
    bbox = cv2.boundingRect(alpha_channel)
    x, y, w, h = bbox
    cropped_img_array = img_array[y:y + h, x:x + w]

    if return_np:
        return cropped_img_array

    if output_path is None:
        raise ValueError("Output path must be provided")

    # Check the file extension of the output path
    _, file_extension = os.path.splitext(output_path)
    if file_format:
        file_extension = file_format if file_format.startswith('.') else '.'+file_format
    else:
        file_extension = file_extension

    # If the output format does not support transparency, convert to BGR
    if file_extension.lower() != '.png':
        cropped_img_array = cv2.cvtColor(cropped_img_array, cv2.COLOR_BGRA2BGR)
    else:
        cropped_img_array = cv2.cvtColor(cropped_img_array, cv2.COLOR_BGRA2RGBA)

    cv2.imencode(file_extension.lower(), cropped_img_array)[1].tofile(output_path)
    return output_path


def apply_mask_and_save_images(input_image, mask, file_path, cropped_images=False):
    """
    Apply a mask to an image and save the image.

    Parameters:
    input_image (str, numpy.ndarray, or PIL.Image): The original image as a file path, a numpy array, or a PIL image object.
    mask (numpy.ndarray): The mask to apply to the image, where non-zero values indicate areas to keep.
    file_path (str): The full file path, including the filename and extension, where the result will be saved.
    cropped_images (bool, optional): If True, crop the image to the area defined by the mask. Defaults to False.
    """

    # Ensure the directory for the file exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if isinstance(input_image, str):
        original_image_np = Image.open(input_image)
    elif isinstance(input_image, np.ndarray):
        original_image_np = Image.fromarray(input_image)
    elif isinstance(input_image, Image.Image):
        original_image_np = input_image
    else:
        raise TypeError("input_image must be a file path, a numpy array, or a PIL Image object")

    # Convert the original image to an RGBA PIL image object
    original_image = Image.fromarray(original_image_np)
    original_image.putalpha(255)  # Ensure the original image has an alpha channel set to opaque

    # Ensure the mask is two-dimensional
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # If it's a 3D array, take the first channel

    # Create a mask image where True corresponds to opaque (255) and False to transparent (0)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), 'L').convert('L')
    mask_image = mask_image.point(lambda p: p > 0 and 255)  # Set non-zero values to 255 for opacity

    # Apply the mask to the original image, making areas outside the mask transparent
    original_image_with_mask = original_image.copy()
    original_image_with_mask.putalpha(mask_image)

    if cropped_images:
        img_array = np.array(original_image_with_mask)
        crop_image_by_alpha_channel(input_image=img_array, output_path=file_path)
    else:
        original_image_with_mask.save(file_path)


def save_boolean_mask(mask: np.ndarray, file_path: str, crop: bool = False, invert: bool = False) -> None:
    """
    Saves a boolean mask as an image file, with options to crop and invert the mask colors.

    Parameters:
      mask (np.ndarray): A boolean mask array with shape (H, W).
      file_path (str): The file path where the image will be saved.
      crop (bool): Whether to crop to the mask's bounding box.
      invert (bool): Whether to invert the mask colors, where True makes the non-masked part black and the masked part transparent.
    """
    # Convert the boolean mask to an 8-bit grayscale image, where True corresponds to 255 and False to 0
    gray_image = Image.fromarray((mask * 255).astype(np.uint8), 'L')

    if crop:
        # Convert the grayscale image to RGBA and add a fully transparent alpha channel
        gray_image_with_alpha = Image.new('RGBA', gray_image.size, (255, 255, 255, 0))
        gray_image_with_alpha.paste(gray_image, mask=gray_image)

        # Convert the PIL image to a numpy array for cropping
        img_array = np.array(gray_image_with_alpha)
        img_array = crop_image_by_alpha_channel(img_array, return_np=True)

        # Convert the cropped numpy array back to a PIL image
        cropped_image = Image.fromarray(img_array, 'RGBA')
    else:
        # Convert the grayscale image to RGBA without cropping
        cropped_image = gray_image.convert('RGBA')

    if invert:
        black_background = Image.new('RGBA', cropped_image.size, (0, 0, 0, 255))
        final_image = Image.alpha_composite(black_background, cropped_image)
    else:
        final_image = cropped_image

    final_image.save(file_path)


def overlay_layer(
        base_path: Optional[str] = None,
        overlay_path: Optional[str] = None,
        output_path: str = 'result_img.png',
        left_margin: Optional[int] = None,
        top_margin: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        canvas_width: Optional[int] = None,
        canvas_height: Optional[int] = None,
        background: Optional[Union[str, list]] = 'transparent',
        rotate_angle: float = 0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        simple_overlay: bool = True
) -> str:
    """
    Overlay an image onto another image with optional adjustments for position, rotation, flipping, or simple overlay.

    Args:
    - base_path: Path to the base image.
    - overlay_path: Path to the overlay image.
    - output_path: Path for the output image.
    - left_margin: Left margin of the overlay image on the base image.
    - top_margin: Top margin of the overlay image on the base image.
    - width: Width of the overlay image.
    - height: Height of the overlay image.
    - canvas_width: Width of the base image. If not specified, original image size is used.
    - canvas_height: Height of the base image. If not specified, original image size is used.
    - background: A string specifying the background color ('transparent', 'white', 'black').
    - rotate_angle: Rotation angle in degrees. Default is 0 (no rotation).
    - flip_horizontal: Whether to horizontally flip the overlay image.
    - flip_vertical: Whether to vertically flip the overlay image.
    - simple_overlay: If True, perform simple overlay without rotation or flipping.

    Returns:
    - Path of the output image.
    """
    if simple_overlay:
        if not (base_path and overlay_path):
            raise ValueError("base_path and overlay_path must be provided for simple overlay")
        base_img = Image.open(base_path)
        overlay_img = Image.open(overlay_path)
        crop_image_by_alpha_channel(overlay_img, output_path=overlay_path)
        base_img.paste(overlay_img, (0, 0), overlay_img)
        base_img.save(output_path)
        return output_path

    elif not (base_path and (canvas_width or canvas_height)):
        raise ValueError("base_path and either canvas_width or canvas_height must be provided")

    if canvas_width is not None and canvas_height is not None:
        default_color = (0, 0, 0, 0)
        if background == 'white':
            base_color = (255, 255, 255, 255)
            base_mode = 'RGB'
        elif background == 'black':
            base_color = (0, 0, 0, 255)
            base_mode = 'RGB'
        else:
            base_color = default_color
            base_mode = 'RGBA' if background == 'transparent' else 'RGB'
        base_img = Image.new(base_mode, (canvas_width, canvas_height), base_color)

    else:
        base_img = Image.open(base_path)

    if not (overlay_path and width and height):
        raise ValueError("overlay_path, width, and height must be provided")

    overlay_img = Image.open(overlay_path)
    crop_image_by_alpha_channel(overlay_img, output_path=overlay_path)
    overlay_img = overlay_img.resize((int(width), int(height)), resample=Image.LANCZOS)
    overlay_img = overlay_img.rotate(rotate_angle, expand=True)  # Set expand=True to avoid cropping

    if flip_horizontal:
        overlay_img = ImageOps.mirror(overlay_img)
    if flip_vertical:
        overlay_img = ImageOps.flip(overlay_img)

    if left_margin is None or top_margin is None:
        raise ValueError("left_margin and top_margin must be provided")

    paste_x = int(left_margin - (overlay_img.width - width) / 2)
    paste_y = int(top_margin - (overlay_img.height - height) / 2)

    base_img.paste(overlay_img, (paste_x, paste_y), overlay_img)

    base_img.save(output_path)

    return output_path


if __name__ == '__main__':
    img1 = "a.png"
    img2 = "b.png"
    output = "result_image.png"
    overlay_layer(
        base_path=img1,
        overlay_path=img2,
        output_path=output,
        simple_overlay=True,
        canvas_width=1024,
        canvas_height=1024,
        background='transparent',
        width=600,
        height=800,
        left_margin=100,
        top_margin=100
    )