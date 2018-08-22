# ColorNet(灰图涂色)
## add color for grayscale image(为灰度图添加色彩)  
[sources from  pavelgonchar(资料来源)：https://github.com/pavelgonchar/colornet](https://github.com/pavelgonchar/colornet)  

[tutorial(教程) http://tinyclouds.org/colorize/](http://tinyclouds.org/colorize/)

The input to the model is the left-side grayscale image. The output is the middle image. The right image is the true color—which the model never gets to see. (输入是左边的灰度图，输出是中间的图像，右边的图像是真实的图像但模型不可见)       
![image](https://wx2.sinaimg.cn/mw1024/8311c72dly1fuiqqzwumuj20im0cik4g.jpg)  

we all knew the grayscale image and the rgb image(我们都知道灰度图和rgb图)，  
now explain the YUV image(现在解释一下YUV图)。  
[https://en.wikipedia.org/wiki/YUV](https://en.wikipedia.org/wiki/YUV)   
YUV is a color encoding system typically used as part of a color image pipeline.(YUV是一种颜色编码系统，通常被用作彩色图像管道的一部分。)  

First order of business was picking a pre-trained model to use(首先是选一个预训练模型来使用)，  
chose the VGG-16 model because it has a simple architecure yet still competitive(选择VGG-16模型因为它有完整的简单架构)。  
![image](https://wx2.sinaimg.cn/mw1024/8311c72dly1fuiqve0u26j20fj0g93yn.jpg)  

Rather than reconstructing the entire color RGB image, I trained models to produce two color channels which I concatenate with the grayscale input channel to produce a YUV image.(比起重建一个完整的rgb图，我选择训练模型生成两个色道，从而与灰度图的输入色道连接在一起来生成一个YUV图像)。     

![image](https://wx1.sinaimg.cn/mw1024/8311c72dly1fuirewjs2mj20gc09nmx5.jpg)  

#### Loss(计算损失)   
The most obvious loss function is a Euclidean distance function between the network RGB output image and the true color RGB image. This works to some extent but I found models converged faster when I used a more complex loss function.(最突出的损失函数是用欧式距离计算网络生成图像与真实图像。这对于很多情况都可行，但我发现当我使用更复杂的损失函数时，模型更快地收敛。)    

Blurring the network output and the true color image and doing Euclidean distance seems to give the gradient decent help. I ended up averaging the normal rgb distance and two blur distances with 3 and 5 pixel gaussian kernels.(欧式距离计算网络生成图像与真实图像，看似可以帮助梯度下降。我最终用3像素和5像素高斯核平均正常的rgb距离和两个色道距离。)  

Also I only calculuate the distance in UV space.(当然我只计算YUV图的UV道)  

##### Network Architecture(网络架构)  
I use ReLUs as activation functions throughout except at the last output to UV channels—there I use a sigmoid to squash values between 0 and 1.(除了最后一层输出UV道，使用sigmoid产生0到1的值外，其他层我使用了ReLUs作为激活函数)  

I use batch norm (BN) instead of bias terms behind every convolution(在每一次卷积后，我使用batch_norm代替偏置项)  

Unlike in classification models there is no max pooling. I need the output to be at full 224 x 224 resolution(不同于分类模型，这里没有max pooling，我需要输出完整的224 x 224尺寸的结果)  

The model forwards a grayscale image thru VGG16 and then using the highest layer infers some color information. Then it upscales the color guess and adds in information from the next highest layer, and so on working down to the bottom of the VGG16 until there is 224 x 224 x 3 tensor.(模型让一张灰度图通过VGG16结构，然后用更高的层得到颜色的信息。然后它更新颜色的预测并加入下一层的信息当中，最后运行到VGG16的最后一层直至其产生了244 x 244 x 3 的张量，也即结果)  
![image](https://wx2.sinaimg.cn/mw1024/8311c72dly1fuikwv7iavj20c90jm3yv.jpg)  

