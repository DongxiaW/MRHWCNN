
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add
from models.DWT import DWT_Pooling, IWT_UpSampling

def unetWavelet(input_size = (400,400,1)):

    def down_block(input_layer, filters, kernel_size=(3,3), activation="relu"):
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(input_layer)
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        pool, theta = DWT_Pooling()(output)
        return output, pool, theta


    def up_block(input_layer, theta, residual_layer, filters, kernel_size=(3,3),activation="relu"):
        output = IWT_UpSampling()(input_layer, theta)
        output = Add()([residual_layer,output])
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        output = Conv2D(filters*2, kernel_size, padding="same", activation=activation)(output)
        return output

    inputs = Input(shape = input_size)

    down1, pool1, theta1 = down_block(inputs,32)
    down2, pool2, theta2 = down_block(pool1,64)
    down3, pool3, theta3 = down_block(pool2,128)
    down4, pool4, theta4 = down_block(pool3,256)

    down5 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation ="relu")(pool4)
    down5 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation ="relu")(down5)
    down5 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation ="relu")(down5)

    up = up_block(down5,theta4,down4,256)
    up = up_block(up,theta3,down3,128)
    up = up_block(up,theta2,down2,64)
    up = up_block(up,theta1,down1,32)

    output = Conv2D(filters=input_size[2], kernel_size=(1, 1), padding="same")(up)
    model = Model(inputs, output)
    print(model.summary()) 
    return model
