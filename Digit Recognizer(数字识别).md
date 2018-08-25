# Digit Recognizer(数字识别)  

[kaggle：https://www.kaggle.com/c/digit-recognizer](https://www.kaggle.com/c/digit-recognizer)  

[kernal-tutorial(教程)：https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)   

## Goal(任务)
The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is.(比赛任务是识别手写的单一数字)  
For every ImageId in the test set, you should predict the correct label.(对于训练集的每个图片ID，应该预测其正确标签)  

## Metric(评估)
This competition is evaluated on the categorization accuracy of your predictions (the percentage of images you get correct).  
(用分类正确率进行评估，正确分类的图片的占比)  

## Introduction to CNN Keras - Acc 0.997 (top 8%)(介绍CNN-keras)  
### 1. Introduction(介绍)  
### 2. Data preparation(数据预处理)   
#### 2.1 Load data(加载数据)  
#### 2.2 Check for null and missing values(查空值和缺省值)  
#### 2.3 Normalization(标准化)  
#### 2.4 Reshape(调整尺寸)  
#### 2.5 Label encoding(标签编码)  
#### 2.6 Split training and valdiation set(切分训练集和验证集)
### 3. CNN
#### 3.1 Define the model(建模)
#### 3.2 Set the optimizer and annealer(优化算法和退火)
#### 3.3 Data augmentation(数据增强)
### 4. Evaluate the model(模型评估)
#### 4.1 Training and validation curves(训练和验证曲线图)
#### 4.2 Confusion matrix(混淆矩阵)
### 5. Prediction and submition(预测和提交)
#### 5.1 Predict and Submit results(预测并提交结果)  

![image](https://wx2.sinaimg.cn/mw1024/8311c72dly1fuly8kab6cj20eq089aah.jpg)   

> import seaborn as sns  
sns.countplot(Y_train)  

![image](https://wx1.sinaimg.cn/mw1024/8311c72dly1fulyalxuukj20cz07pglm.jpg)   

> plt.imshow(X_train[0][:,:,0])  

![image](https://wx2.sinaimg.cn/mw1024/8311c72dly1fulycfta5pj208t072t8n.jpg)  

## CNN

> model = Sequential()  
model.add(Conv2D(filters = 32,kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))  
model.add(Conv2D(filters = 32,kernel_size = (5,5),padding = 'Same', activation ='relu'))  
model.add(MaxPool2D(pool_size=(2,2)))  
model.add(Dropout(0.25))  
model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = 'Same', activation ='relu'))  
model.add(Conv2D(filters = 64,kernel_size = (3,3),padding = 'Same', activation ='relu'))  
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))  
model.add(Dropout(0.25))  
model.add(Flatten())  
model.add(Dense(256, activation ="relu"))  
model.add(Dropout(0.5))  
model.add(Dense(10, activation = "softmax"))  

> optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)  

> model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])  

In order to make the optimizer converge faster and closest to the global minimum of the loss function, i used an annealing method of the learning rate (LR).(为了使优化算法更快更近地达到代价函数的全局最小，我使用了学习率(LR)的退火方式)  

Its better to have a decreasing learning rate during the training to reach efficiently the global minimum of the loss function.(一种较好的方式是,在训练中到达代价函数的全局最小,用逐渐变小的学习率)    

To keep the advantage of the fast computation time with a high LR, i decreased the LR dynamically every X steps (epochs) depending if it is necessary (when accuracy is not improved).(为了保持高学习率的计算更快的优势，当必要时，即准确率无法提高，我动态地每x步减小学习率)  

With the ReduceLROnPlateau function from Keras.callbacks, i choose to reduce the LR by half if the accuracy is not improved after 3 epochs.(用Keras.callbacks里的ReduceLROnPlateau函数，当准确率无法在3次迭代之后提高，我选择让学习率减半)   

> learning_rate_reduction =ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)  

## Data augmentation(数据增强)  
The improvement is important :(这一步骤很重要)  

Without data augmentation i obtained an accuracy of 98.114%(没有数据增强仅获得准确率98.114%)  

With data augmentation i achieved 99.67% of accuracy(数据增强之后达到准确率99.67%)  

## Evaluate the model(模型评估)  
plot the error classification(观察错误分类样本)  

![image](https://wx1.sinaimg.cn/mw1024/8311c72dly1fulza084ylj20b303yaa2.jpg)  

get the accuracy 0.997(准确率) 