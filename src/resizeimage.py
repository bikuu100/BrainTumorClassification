from PIL import Image
import os
import traceback

def resize_image(image_path, target_size):
    try:
        # Open the image
        image = Image.open(image_path)
        # Resize the image to the target size
        resized_image = image.resize(target_size, Image.LANCZOS)
        return resized_image
    except Exception as e:
        print(f"Error resizing image {image_path}: {e}")
        traceback.print_exc()
        return None

def resize_images_in_directory(input_directory, output_directory, target_size):
    image_count = 0
    failed_images = 0
    # Walk through the input directory
    for root, _, files in os.walk(input_directory):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)
                relative_path = os.path.relpath(image_path, input_directory)
                output_path = os.path.join(output_directory, relative_path)

                # Create the output directory if it doesn't exist
                output_dir = os.path.dirname(output_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    print(f"Created directory {output_dir}")

                # Resize and save the image
                resized_image = resize_image(image_path, target_size)
                if resized_image is not None:
                    # Convert images to RGB if necessary
                    if resized_image.mode in ['RGBA', 'P']:
                        resized_image = resized_image.convert('RGB')
                    resized_image.save(output_path)
                    print(f"Resized and saved {output_path}")
                    image_count += 1
                else:
                    print(f"Failed to resize image {image_path}")
                    failed_images += 1

    print(f"Processed {image_count} images, failed to process {failed_images} images.")

# Define the target size (width, height)
target_size = (256, 256)

# Directories
train_directory = r"data\Train"
resized_train_directory = r"data\resized_train"

test_directory = r"data\Test"
resized_test_directory = r"data\resized_test"

# Resize images in the train directory and save them to the new directory
print(f"Resizing images in {train_directory} and saving to {resized_train_directory}")
resize_images_in_directory(train_directory, resized_train_directory, target_size)

# Resize images in the test directory and save them to the new directory
print(f"Resizing images in {test_directory} and saving to {resized_test_directory}")
resize_images_in_directory(test_directory, resized_test_directory, target_size)
