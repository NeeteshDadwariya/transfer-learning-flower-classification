import pandas as pd
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions


# Note that you can save models in different formats. Some format needs to save/load model and weight separately.
# Some saves the whole thing together. So, for your set up you might need to save and load differently.

def load_model_weights(model, weights=None):
    my_model = tf.keras.models.load_model(model)
    my_model.summary()
    return my_model


def get_images_labels(df, classes, image_size):
    test_images = []
    image_names = []
    test_labels = []
    for index, row in df.iterrows():
        label = row[1].strip()
        image_names.append(row[0])
        img = tf.keras.utils.load_img(row[0], target_size=image_size)
        img_array = tf.keras.utils.img_to_array(img)
        img_preprocessed = preprocess_input(img_array)
        test_images.append(img_preprocessed)
        test_labels.append(classes.index(label))
    return np.array(test_images), np.array(test_labels), image_names


def decode_img(img, img_height, img_width):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trasnfer Learning Test")
    parser.add_argument('--model', type=str, default='my_model.h5', help='Saved model')
    parser.add_argument('--weights', type=str, default=None, help='weight file if needed')
    parser.add_argument('--test_csv', type=str, default='flowers_test.csv', help='CSV file with true labels')

    args = parser.parse_args()
    model_name = args.model
    weights = args.weights
    test_csv = args.test_csv

    test_df = pd.read_csv(test_csv)
    classes = ['astilbe', 'bellflower', 'black-eyed susan', 'calendula', 'california poppy', 'carnation',
               'common daisy', 'coreopsis', 'dandelion', 'iris', 'rose', 'sunflower', 'tulip']

    IMG_SIZE = (224, 224)
    #IMG_SIZE = (160, 160)
    # Rewrite the code to match with your setup
    test_images, test_labels, image_names = get_images_labels(test_df, classes, IMG_SIZE)

    model = load_model_weights(model_name)

    for i, image in enumerate(test_images):
        predictions = model.predict(tf.expand_dims(image, 0))
        print("The image {} most likely belongs to \"{}\" with a {:.2f} percent confidence."
              .format(image_names[i], classes[np.argmax(predictions)], 100 * np.max(predictions)))

    loss, acc = model.evaluate(test_images, test_labels, verbose=2)

    print('Test model, accuracy: {:5.5f}%'.format(100 * acc))
