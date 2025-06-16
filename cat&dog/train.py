import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential

train_datagen = ImageDataGenerator(rescale=1.0/255,#归一化，必须的
                                   rotation_range=30,
                                   width_shift_range=0.2,#表示将图片随机水平平移百分之20。
                                   height_shift_range=0.2,#表示将图片随机垂直平移百分之20。
                                   shear_range=0.2,#表示对图片进行随机裁剪的比例为20%
                                   zoom_range=0.2, #表示对图片进行随机缩放的比例为20%
                                   horizontal_flip=True, #表示对图片水平反转
                                   fill_mode='nearest' #表示填充方式为最近邻填充
                                   )
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_data = train_datagen.flow_from_directory(directory='../train',
                                               target_size=(224, 224),#表示图片处理后的像素大小
                                               color_mode='rgb',
                                               # batch_size=50,
                                               class_mode='sparse'# 表示标签形式，categorical表示独热编码形式,binary表示2分类,sparse表示整数标签
                                               )
# train_data下方有如下内置函数：
# 1、.samples用于查看图片的总个数
# 2、.class_indics用于查看图片的标签和名称，以字典形式保存
# 3、.image_shape用于查看图片的形状
# 4、.__next__()用于查看一批次读取的图片数据
x,y = train_data.__next__()
print(x.shape)
print(y.shape)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())   #添加展平层
model.add(Dense(units=512, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=256, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=2, activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_data, epochs=10,verbose=1,batch_size=50)

model.save("cat&dog.h5")

plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('Model Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'orange')
plt.title('Model Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()