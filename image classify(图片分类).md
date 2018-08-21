# image classify(图片分类)
## CNN classify dogs and cats(猫狗二分类)
[Tutorial(教程):https://developers.google.com/machine-learning/practica/image-classification](https://developers.google.com/machine-learning/practica/image-classification)

**the training samples are just only two kinds:dog and cat(训练集只有两种图片：猫和狗)**  
![image](https://wx2.sinaimg.cn/mw1024/8311c72dly1fuhculbeevj20q10ch157.jpg)  
==1、basic CNN model with tensorflow.keras(简单的CNN模型)==

> img_input = layers.Input(shape=(150, 150, 3))  
> x = layers.Conv2D(16, 3,activation='relu')(img_input)  
> x = layers.MaxPooling2D(2)(x)  
> x = layers.Conv2D(32, 3, activation='relu')(x)  
> x = layers.MaxPooling2D(2)(x)  
> x = layers.Conv2D(64, 3, activation='relu')(x)  
> x = layers.MaxPooling2D(2)(x)  
> x = layers.Flatten()(x)  
> x = layers.Dense(512, activation='relu')(x)  
> output = layers.Dense(1, activation='sigmoid')(x)  
> model = Model(img_input, output)

**the network architecture(网络结构)**    
![image](https://wx1.sinaimg.cn/mw1024/8311c72dly1fuhd5mukzwj20f20e7dft.jpg)

**using RMSprop for optimization algorithm(优化算法采用RMSprop)，  
and there also use Data Augmentation(也用到数据增强)，  
cause the training samples are not enough and it can help the network learn better(训练数据不足且有助于网络学得更好)。**   
> from tensorflow.keras.preprocessing.image import ImageDataGenerator

**Visualizing Intermediate Representations of erery layer of the CNN(可视化中间过程)**
![image](https://wx3.sinaimg.cn/mw1024/8311c72dly1fuhdj6sgtrj20s70dr147.jpg)

==2、Reducing Overfitting(减缓过拟合)==  
**Data Augmentation(数据增强)**  
> datagen = ImageDataGenerator(  
>&nbsp;&nbsp;rotation_range=40,  
>&nbsp;&nbsp;width_shift_range=0.2,  
>&nbsp;&nbsp;height_shift_range=0.2,  
>&nbsp;&nbsp;shear_range=0.2,  
>&nbsp;&nbsp;zoom_range=0.2,  
>&nbsp;&nbsp;horizontal_flip=True,  
>&nbsp;&nbsp;fill_mode='nearest')  

**Adding Dropout(使用dropout)**  
> x = layers.Dropout(0.5)(x)

==3、Feature Extraction and Fine-Tuning(迁移学习和微调)==  
**use the Inception V3 model developed at Google(使用Inception V3模型),  
and pre-trained on ImageNet(已经预训练好的ImageNet)。**   
> from tensorflow.keras.applications.inception_v3 import InceptionV3  
> local_weights_file ='/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'  
> pre_trained_model = InceptionV3( input_shape=(150, 150, 3),   include_top=False, weights=None)  
> pre_trained_model.load_weights(local_weights_file)  

**won't update the weights of the pretrained model during training(固定前面网络层的参数(权值))**  
> for layer in pre_trained_model.layers:  
>&nbsp;&nbsp;layer.trainable = False  

**let's stick a fully connected classifier on top of last_output(添加属于自己的网络层用于输出)**  
> last_output = pre_trained_model.get_layer('mixed7').output  
x = layers.Flatten()(last_output)  
x = layers.Dense(1024, activation='relu')(x)  
x = layers.Dropout(0.2)(x)  
x = layers.Dense(1, activation='sigmoid')(x)  
model = Model(pre_trained_model.input, x)  

**All we need to do to implement fine-tuning is to set the top layers of Inception V3 to be trainable, recompile the model (necessary for these changes to take effect), and resume training. Let's unfreeze all layers belonging to the mixed7 module—i.e., all layers found after mixed6—and recompile the model:(迁移学习的网络开放前面更多网络层进行训练，更好地拟合数据)**  
> for layer in pre_trained_model.layers:  
> &nbsp;&nbsp;if unfreeze:  
>&nbsp;&nbsp;&nbsp;&nbsp;layer.trainable = True  
>&nbsp;&nbsp;if layer.name == 'mixed6':  
>&nbsp;&nbsp;&nbsp;&nbsp;unfreeze = True  

==finished！！！==