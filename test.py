from glob import glob
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from models.unet import unet
from models.unetWavelet import unetWavelet
import cv2 as cv
import os

from numpy.random import seed
seed(1)

def add_motion_blur(img):
    kernels = glob('blur_kernels_20/*.png')
    idx = np.random.randint(0, len(kernels))

    kernel = plt.imread(kernels[idx])
    kernel = kernel / np.sum(kernel)  # normalize

    blurred = cv.filter2D(img, -1, kernel, borderType=cv.BORDER_REPLICATE)

    return blurred

def display_img(noisy_image, residual_image):
    # Display result image
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.imshow(noisy_image)
    plt.title('Noisy image')
    ax2 = fig.add_subplot(2,1,2)
    ax2.imshow(noisy_image-residual_image)
    plt.title('Denoised image')
    plt.show()


def test(FLAGS):

    ### Parameters of the testing
    #cropShape = (192,192,3)
    cropShape = (64,64,3)
    test_path = FLAGS.test_path
    #noise_level = FLAGS.noise
    architecture = FLAGS.architecture
    filepath = FLAGS.load_weights_path
    sliding_window = FLAGS.sliding_window

    # Load model
    if architecture == "unet":
        model = unet(cropShape)
    elif architecture == "wavelet":
        model = unetWavelet(cropShape)
    else : 
        raise RuntimeError('Unkwown architecture, please choose from "unet" or "wavelet".')
    model.load_weights(filepath)

    # Load the images in the test folder
    test_files =[img_path for img_path in glob(test_path + '/*.png')]
    i = 1
    for img_path in test_files:
        image = load_img(img_path) 
        image = img_to_array(image)
        
        if sliding_window != 0:
            image = image[:sliding_window*((image.shape[0]-cropShape[0])//sliding_window)+cropShape[0], :sliding_window*((image.shape[1]-cropShape[1])//sliding_window)+cropShape[1]]
        else:
            image = image[:(image.shape[0]//cropShape[0])*cropShape[0], :(image.shape[1]//cropShape[1])*cropShape[1]]
        
        ## Add noise
        #noise = np.random.normal(0,noise_level,image.shape)
        #noisy_image = image + noise
        #noisy_image /= 255.0
        #residual_image = np.empty(image.shape)

        blurred_image = add_motion_blur(image)
        blurred_image /= 255.0
        residual_image = np.empty(image.shape)

        # Divide the image into subimages of size cropShape to match the model's input size.
        if sliding_window!=0:
            sum_img = np.zeros(image.shape)
            count_img = np.zeros(image.shape)
            
            for y in range(0,sliding_window+image.shape[0]-cropShape[0],sliding_window):
                for x in range(0,sliding_window+image.shape[1]-cropShape[1],sliding_window):
                    extract = blurred_image[y:y+cropShape[0],x:x+cropShape[1]]
                    sum_img[y:y+cropShape[0],x:x+cropShape[1]] += model.predict(extract[np.newaxis,...])[0]
                    count_img[y:y+cropShape[0],x:x+cropShape[1]]+= 1

            residual_image = sum_img/count_img
        else:
            for block_y in range(0,image.shape[0]//cropShape[0]):
                for block_x in range(0,image.shape[1]//cropShape[1]):
                    extract = blurred_image[block_y*cropShape[0]:(block_y+1)*cropShape[0],block_x*cropShape[1]:(block_x+1)*cropShape[1]]
                    residual_image[block_y*cropShape[0]:(block_y+1)*cropShape[0],block_x*cropShape[1]:(block_x+1)*cropShape[1]] = model.predict(extract[np.newaxis,...])[0]

        #normalize_img = np.zeros(noisy_image.shape)
        #normalize_img = 
        # Display the noised and denoised images
        #display_img(noisy_image,residual_image)
        cv.imwrite('wavelet_rot05_blur_20/blurred/'+str(i)+'.png',blurred_image*255)
        cv.imwrite('wavelet_rot05_blur_20/reconstructed/'+str(i)+'.png',(blurred_image-residual_image)*255)
        cv.imwrite('wavelet_rot05_blur_20/original/'+str(i)+'.png',image)
        #np.save('unet_noisy/original/'+str(i)+'.npy',noisy_image)
        #np.save('unet_noisy/reconstructed/'+str(i)+'.npy',residual_image)
        i += 1



