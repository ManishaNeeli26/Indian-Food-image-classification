# Indian-Food-image-classification
import os
os.listdir("INDIAN_FOOD/Indian Food Images/Indian Food Images/")
Items=['adhirasam',
 'aloo_gobi',
 'aloo_matar',
 'aloo_methi',
 'aloo_shimla_mirch',
 'aloo_tikki',
 'anarsa',
 'ariselu',
 'bandar_laddu',
 'basundi',
 'bhatura',
 'bhindi_masala',
 'biryani',
 'boondi',
 'butter_chicken',
 'chak_hao_kheer',
 'cham_cham',
 'chana_masala',
 'chapati',
 'chhena_kheeri',
 'chicken_razala',
 'chicken_tikka',
 'chicken_tikka_masala',
 'chikki',
 'daal_baati_churma',
 'daal_puri',
 'dal_makhani',
 'dal_tadka',
 'dharwad_pedha',
 'doodhpak',
 'double_ka_meetha',
 'dum_aloo',
 'gajar_ka_halwa',
 'gavvalu',
 'ghevar',
 'gulab_jamun',
 'imarti',
 'jalebi',
 'kachori',
 'kadai_paneer',
 'kadhi_pakoda',
 'kajjikaya',
 'kakinada_khaja',
 'kalakand',
 'karela_bharta',
 'kofta',
 'kuzhi_paniyaram',
 'lassi',
 'ledikeni',
 'litti_chokha',
 'lyangcha',
 'maach_jhol',
 'makki_di_roti_sarson_da_saag',
 'malapua',
 'misi_roti',
 'misti_doi',
 'modak',
 'mysore_pak',
 'naan',
 'navrattan_korma',
 'palak_paneer',
 'paneer_butter_masala',
 'phirni',
 'pithe',
 'poha',
 'poornalu',
 'pootharekulu',
 'qubani_ka_meetha',
 'rabri',
 'rasgulla',
 'ras_malai',
 'sandesh',
 'shankarpali',
 'sheera',
 'sheer_korma',
 'shrikhand',
 'sohan_halwa',
 'sohan_papdi',
 'sutar_feni',
 'unni_appam']
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen=ImageDataGenerator(rescale=1/255.)
train_dir="INDIAN_FOOD/Indian Food Images/Indian Food Images/"
train_data=datagen.flow_from_directory(train_dir,batch_size=32,seed=42,class_mode="categorical",target_size=(256,256))
len(train_data)
input_shape1=(1,256,256,3)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(80, activation='softmax'))
model.summary()
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])
model.fit(train_data,epochs=10,steps_per_epoch=len(train_data))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img=mpimg.imread("chapathi.jpeg")
img.shape
plt.imshow(img)
from PIL import Image
img_1=Image.open("chapathi.jpeg")
img_rescaled_size=img_1.resize(size=(256,256))
import numpy as np
img_rescale=np.asarray(img_rescaled_size)/255.
img_rescale.shape
p1=model.predict(img_rescale[np.newaxis,...])
item=np.argmax(p1)
plt.imshow(img_1)
Items[item]
def image_name_result():
    img=Image.open("#imagefileaddress")
    img_reshaped=img.resize(size=(256,256))
    img_rescale=np.asarray(img_reshaped)/255
    img_predict=model.predict(img_rescale[np.newaxis,...])
    item=np.argmax(img_predict)
    print(Items[item])
