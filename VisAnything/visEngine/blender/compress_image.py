import os
from PIL import Image


def compress_image(input_path, output_path, quality=85):
    """
    Compress the image and save it to the output path.

    :param input_path: Path to the input image.
    :param output_path: Path to save the compressed image.
    :param quality: Quality of the compressed image (1-100).
    """
    with Image.open(input_path) as img:
        # Convert RGBA images to RGB
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.save(output_path, 'JPEG', quality=quality)


def process_directory(directory, quality=85):
    """
    Process all images in the given directory and save the compressed images
    in a new directory with "_compressed" suffix.

    :param directory: Path to the directory to process.
    :param quality: Quality of the compressed images (1-100).
    """
    # Create the output directory
    output_directory = f"{directory}_compressed"
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_directory, file)

                # Compress the image
                compress_image(input_path, output_path, quality)
                print(f"Compressed {input_path} and saved to {output_path}")


def main(input_path, quality=85):
    """
    Main function to process the input path.

    :param input_path: Path to the directory containing subdirectories with images.
    :param quality: Quality of the compressed images (1-100).
    """
    # Iterate over all subdirectories in the input path
    for item in os.listdir(input_path):
        item_path = os.path.join(input_path, item)
        if os.path.isdir(item_path):
            process_directory(item_path, quality)


if __name__ == "__main__":
    input_path = input("Enter the path to the directory: ")
    quality = int(input("Enter the quality for compression (1-100): "))
    main(input_path, quality)
