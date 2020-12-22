
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate

def unet(input_size = (400,400,1)):

    def down_block(input_layer, filters, kernel_size=(3,3), activation="relu"):
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(input_layer)
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        return output, MaxPooling2D(pool_size=(2,2))(output)


    def up_block(input_layer, residual_layer, filters, kernel_size=(3,3),activation="relu"):
        output = UpSampling2D(size = (2,2))(input_layer)
        output = Concatenate(axis = 3)([residual_layer,output])
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        output = Conv2D(filters, kernel_size, padding="same", activation=activation)(output)
        return output

    inputs = Input(shape = input_size)

    down1, pool1 = down_block(inputs,32)
    down2, pool2 = down_block(pool1,64)
    down3, pool3 = down_block(pool2,128)
    down4, pool4 = down_block(pool3,256)

    down5 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation ="relu")(pool4)
    down5 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation ="relu")(down5)

    up = up_block(down5,down4,256)
    up = up_block(up,down3,128)
    up = up_block(up,down2,64)
    up = up_block(up,down1,32)

    output = Conv2D(filters=input_size[2], kernel_size=(1, 1), padding="same")(up)
    model = Model(inputs, output)
    print(model.summary())
    return model
