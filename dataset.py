import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical


class DigitsDatasetTF:
    def __init__(self, directory):
        self.directory = directory
        self.filenames = os.listdir(directory)
        self.image_size = (46, 160)  # Resize target

    def preprocess_image(self, image_path):
        image = load_img(image_path, target_size=self.image_size, color_mode="rgb")
        image = img_to_array(image)
        image = image / 255.0  # Normalize to [0, 1]
        return image

    def parse_label(self, filename):
        label_part = os.path.splitext(filename)[0][:4]  # Extract label from filename
        label = [int(ch) for ch in label_part]
        label = label.join()
        return np.array(label, dtype=np.int32)

    def __call__(self):
        def generator():
            for filename in self.filenames:
                image_path = os.path.join(self.directory, filename)
                image = self.preprocess_image(image_path)
                label = self.parse_label(filename)
                yield image, label

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=(tf.float32, tf.int32),
            output_shapes=(
                [self.image_size[0], self.image_size[1], 3],
                [None],
            ),  # Adjust based on your needs
        )
        return dataset


# Usage
directory = "/content/drive/MyDrive/Deep Learning/data/rescator_1/train"
dataset = DigitsDatasetTF(directory)()
# dataset = dataset.batch(32)  # Example of batching


def preprocess_labels(image, label):
    label = tf.one_hot(
        label, depth=num_classes
    )  # Adjust if your labels need one-hot encoding
    return image, tf.reshape(label, [-1])  # Ensure label shape is correct


# Assuming your dataset and preprocessing are correctly set up
dataset = dataset.map(preprocess_labels)  # Apply any necessary preprocessing
dataset = dataset.batch(32)  # Batch the dataset

# Ensure the dataset shapes are correct
print(dataset.element_spec)

# Split dataset into training and validation (adjust split ratio as needed)
train_size = int(0.8 * len(os.listdir(directory)))
val_size = len(os.listdir(directory)) - train_size
train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)
