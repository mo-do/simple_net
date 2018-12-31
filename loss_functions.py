from keras import backend as K

def my_categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)