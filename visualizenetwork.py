import argparse
import cv2
import h5py
import keras.backend as K
from keras.models import load_model
from keras import __version__ as keras_version
import matplotlib.pyplot as plt
import numpy as np
import os

# My file
import common

# Returns the outputs from a layer.
def get_outputs_from_layer(model, layer_name, img):
    layer = next(layer for layer in model.layers if layer.name == layer_name)
    output_tensor = layer.output

    inp = model.inputs
    
    print('is list: {}'.format(isinstance(inp, list)))

    func = K.function(inp + [K.learning_phase()], [layer.output])

    input_value = np.expand_dims(img, axis=0)
    output_res = func([input_value] + [0])
    print('output_res length: {}'.format(len(output_res)))
    print('output[0] shape: {}'.format(output_res[0].shape))
    
    output_res = np.squeeze(output_res)
    print('output_res length: {}'.format(len(output_res)))
    print('output[0] shape: {}'.format(output_res[0].shape))
    print('output shape: {}'.format(output_res.shape))

    return output_res

# Takes a list of float values and rescales to 0 -- 255 uint8
def normalize_to_byte(img):
    max = img.max()
    min = img.min()
    img = (img + min)/(max - min)*255
    img = img.astype(np.uint8)
    return img


def show_layer_outputs(outputs):
    fig = plt.figure()

    for i in range(outputs.shape[-1]):
        output = outputs[..., i]
        output = normalize_to_byte(output)
        ax = fig.add_subplot(len(outputs), 1, i+1)
        ax.imshow(output)


def save_layer_outputs(outputs, savedir):
    for i in range(outputs.shape[-1]):
        output = outputs[..., i]
        if i == 0:
            print(output)
        output = normalize_to_byte(output)
        cv2.imwrite(os.path.join(savedir, '{}.jpg'.format(i)), output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'layer',
        type=str,
        help='The name of the layer.'
    )
    parser.add_argument(
        'image',
        type=str,
        help='Path to image image.'
    )
    parser.add_argument(
        '--todir',
        type=str,
        nargs='?',
        help='Directory to save activation outputs.'
    )
    args = parser.parse_args()

    # check that model Keras version is same as local Keras version
    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

    img = cv2.imread(args.image)

    img = common.preprocess_img(img)

    outputs = get_outputs_from_layer(model, args.layer, img)
    
    if args.todir:
        save_layer_outputs(outputs, args.todir)
    else:
        show_layer_outputs(outputs)

