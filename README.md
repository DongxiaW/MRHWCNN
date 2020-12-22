# MRHWCNN

## Author: Dongxia Wu

### Description
In this project, I developed the Multi-level Rotation Haar wavelet Convolutional Neural Network for image deblurring. It is implemented by Keras.

## Table of contents

- [Requirements](#requirements)
- [Preprorcessing](#preprocessing)
- [How to run](#run)
- [Reference](#reference)

## Requirements <a name="requirements"></a>
* tensorflow 2.3.1
* numpy 1.18.1
* keras 2.4.3

## Preprocessing <a name="preprocessing"></a>
Download DIV2K_train_LR_x8 and DIV2K_valid_LR_x8 from https://data.vision.ee.ethz.ch/cvl/DIV2K/ \
Move them to the folder DIV2K.

## How to run  <a name="run"></a>

Run the following code for training and test:
```
python3 init.py -tr DIV2K/DIV2K_train_HR/ -t DIV2K/DIV2K_valid_HR/ -a wavelet -m train
python3 init.py -t DIV2K/DIV2K_valid_HR/ -a wavelet -m test -lw weights/DenoisingWavelet.h5
```

## Reference  <a name="reference"></a>
Liu P, Zhang H, Lian W, et al. Multi-level wavelet convolutional neural networks[J]. IEEE Access, 2019, 7: 74973-74985.(https://ieeexplore.ieee.org/abstract/document/8732332) \
Keras implementation of MWCNN (https://github.com/AureliePeng/Keras-WaveletTransform)
