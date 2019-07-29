#### 一、基础操作

##### 展示图片：

```python
import cv2

filepath = 'Dilireba/1.jpg'

image = cv2.imread(filepath)
gray = gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(image, (7,7), 0)

cv2.imshow("Original Image", image)
cv2.imshow("Gray Image", gray_image)
cv2.imshow("Blurred Image", blurred_image)

cv2.waitKey(0)
```



##### 边缘检测：

最流行的边缘检测函数由Canny完成：

```python
canny = cv2.Canny(blurred_image, 10, 30)
cv2.imshow("Canny with low thresholds", canny)

canny2 = cv2.Canny(blurred_image, 50, 150)
cv2.imshow("Canny with high thresholds", canny2)
```



##### 从文件播放视频：

```python
import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```



##### 保存视频

保存视频需要创建**VideoWriter**对象，需指定**FourCC**编码、帧率、帧大小和**isColor**标识等。 **FourCC**是一个用于指定视频编解码器的4字节编码，完整了列表可见[fourcc.org](www.fourcc.org)，与平台相关，就作者已测试的Frdora而言有：DIVX、XVID、MJPG、X264、WMV1、WMV2。 其中更偏好XVID， MJPG会产生很大的视频， X264的视频则较小。

```python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
```



#### 二、图像处理

##### 1.改变颜色空间

OpenCV中有150种颜色空间转换方法，这里介绍最常用的BGR$\leftrightarrow$Gray以及BGR$\leftrightarrow$HSV之间的转换。使用函数`cv2.cvtColor(imput_image, flag)`实现颜色转换，其中`flag`决定转换类型，`cv2.COLOR_BGR2GRAY`为BGR$\rightarrow$GRAY，`cv2.COLOR_BGR2HSV`为BGR$\rightarrow$HSV。

```python
import cv2
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
```

HSV为色相、饱和度、明度(Value)。分别代表：

- 色相（H）是色彩的基本属性，就是平常所说的颜色名称，如红色、黄色等。
- 饱和度（S）是指色彩的纯度，越高色彩越纯，低则逐渐变灰，取0-100%的数值。
- 明度（V），亮度（L），取0-100%。

HSV是RGB色彩模型中的点在圆柱坐标系中的表示法，比基于直角坐标系的RGB的集合结构更加直观。这个圆柱的中心轴取值为自底部的黑色到顶部的白色，而在它们中间的是灰色，绕这个轴的角度对应于“色相”，到这个轴的距离对应于“饱和度”，而沿着这个轴的高度对应于“亮度”，“色调”或“明度”。

对HSV，OpenCV中色相范围为[0,179]，饱和度范围为[0,255]，明度范围为[0,255]。不同的软件会有不同的值域，因此当比较时需做比较时，需要规范化这些范围。

在将BGR图像转换为HSV后，就可以用其来提取出特定颜色的物体。HSV比RGB更易于表示颜色。

```python
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while(1):
    # Take each frame
    _, frame = cap.read()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
```

要找到BGR对应的HSV值，也可以使用`cv2.cvtColor()`函数：

```python
green = np.uint8([[[0,255,0 ]]])
hsv_green = cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
```



##### 2.几何变换

OpenCV提供



##### 3.图像阈值

**简单阈值**：若像素值大于某个数，则赋为1,否则为0。OpenCV提供了`cv2.threshold`函数，地一个参数是灰度图，第二个参数是阈值，第三个为`maxVal`，表示若像素大于阈值时被赋予的值，第四个参数表示阈值类型：

- cv2.THRESH_BINARY
- cv2.THRESH_BINARY_INV
- cv2.THRESH_TRUNC
- cv2.THRESH_TOZERO
- cv2.THRESH_TOZERO_INV

有两个输出，第一个是`retval`，第二个是阈值后的图像。

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('gradient.png',0)
ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

plt.show()
```

**自适应阈值**：前面的是全局阈值，而自适应阈值只为图像的小块区域计算阈值，这样不同区域就会得到不同的阈值，这样光照变化的图像能得到更好的结果。`cv2.adaptiveThreshold()`函数输出一个结果，有3个特殊参数：

- `Adaptive Method`，决定如何计算阈值：
  - `cv2.ADAPTIVE_THRESH_MEAN_C`：阈值是临近区域的均值；
  - `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`：阈值是临近区域的加权和，而权值是高斯窗口；
- `Block Size`：临近区域的大小；
- `C`：从均值或加权均值减去的常数。

```python
img = cv2.imread('dave.jpg',0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
```

**Otsu二值化**：要判断一个全局阈值的优劣，就需要试错。考虑双峰图像（直方图有两个高峰），可以取两峰均值为阈值。这就是Ostu二值化的原理。在使用`cv2.threshold()`函数时传递`cv2.THRESH_OSTU`标识，阈值为0，之后算法就会找到最优阈值并将其返回为前面提到的`retVal`。在执行时，算法会找到最小化加权类内方差：
$$
\sigma_w^2(t) = q_1(t)\sigma_1^2(t)+q_2(t)\sigma_2^2(t)
$$
其中：
$$
q_1(t) = \sum_{i=1}^{t} P(i) \quad \& \quad q_1(t) = \sum_{i=t+1}^{I} P(i)
\\
\mu_1(t) = \sum_{i=1}^{t} \frac{iP(i)}{q_1(t)} \quad \& \quad \mu_2(t) = \sum_{i=t+1}^{I} \frac{iP(i)}{q_2(t)}
\\
\sigma_1^2(t) = \sum_{i=1}^{t} [i-\mu_1(t)]^2 \frac{P(i)}{q_1(t)} \quad \& \quad \sigma_2^2(t) = \sum_{i=t+1}^{I} [i-\mu_1(t)]^2 \frac{P(i)}{q_2(t)}
$$
其实就是找一个位于双峰之间的值以使两类的方差最小。可以用python实现为：

```python
img = cv2.imread('noisy2.png',0)
blur = cv2.GaussianBlur(img,(5,5),0)

# find normalized_histogram, and its cumulative distribution function
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
thresh = -1

for i in xrange(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights

    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i

# find otsu's threshold value with OpenCV function
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print thresh,ret
```

下面是一个示例：

```python
img = cv2.imread('noisy2.png',0)

# global thresholding
ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img,(5,5),0)
ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

for i in xrange(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
```



##### 4.图像平滑

**2D卷积（图像滤波）**：图像可用低通滤波器(LPF)过滤，以消除噪声或模糊图像；也可用高通滤波器，用于寻找边缘。OpenCV中提供了`cv2.filter2D()`函数以用用图像卷积一个核。使用卷积核：
$$
K =  \frac{1}{25} \begin{bmatrix} 1 & 1 & 1 & 1 & 1  \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \\ 1 & 1 & 1 & 1 & 1 \end{bmatrix}
$$
卷积图像的示例代码：

```python
img = cv2.imread('opencv_logo.png')
kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()
```

**图像模糊（图像平滑）**：图像模糊由低通滤波核卷积图像实现，有助于去噪。实际去除图像中的高频内容（噪声、边等），从而使边变得模糊。OpenCV中主要提供了4种模糊技术，分别为平均、高斯、中值、双边等：

- **平均**：使用正规化滤波器完成，取滤波器下所有像素的均值代替中心值，使用函数`cv2.blur()`或`cv2.boxFilter()`完成；若不希望使用箱式滤波器，使用`cv2.boxFilter()`并传递参数`normalize=False`；

  ```python
  img = cv2.imread('opencv_logo.png')
  blur = cv2.blur(img,(5,5))
  plt.subplot(121),plt.imshow(img),plt.title('Original')
  plt.xticks([]), plt.yticks([])
  plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
  plt.xticks([]), plt.yticks([])
  plt.show()
  ```

- **高斯滤波**：使用高斯核，由函数`cv2.GaussianBlur()`实现，需指定其核的宽度和高度且为正奇数，还需分别指定X和Y方向上的偏差`sigmaX`和`sigmaY`，若仅指定`sigmaX`，`sigmaY`与其相同，若都赋为0，则由核大小计算获得。

  ```python
  blur = cv2.GaussianBlur(img,(5,5),0)
  ```

- **中值滤波**：将卷积窗口下的像素中位数替代中心值，函数为`cv2.medianBlur()`，在去除盐椒噪声上特别有效，且替代的值必然存在于原始图像中，这就有效减小了噪声。核大小需为正奇数。

  ```python
  median = cv2.medianBlur(img, 5)
  ```

- **双边过滤**：前面的滤波器都会模糊边缘，但双边滤波器则在高效去噪的同时保留边缘。高斯滤波器仅与空间相关，只考虑周围像素然后求出高斯加权均值，而不管像素是否在边缘。双边滤波器也在空域使用高斯滤波器，但还使用一个为像素密度差异函数的（乘法）高斯成分。空间高斯函数确保仅考虑空间相邻的像素来滤波，而密度域的高斯成分则确保与中心密度相似的像素包含于模糊密度值的计算。

  ```python
  blur = cv2.bilateralFilter(img,9,75,75)
  ```



##### 5.形态变换

形态变换是基于图像形状的简单操作，通常在二值图像上操作。需要两个输入，原始图像，以及决定操作本质的结构元素或核。两种基本操作是腐蚀(erosion)和膨胀(dilation)；然后是其变体形式打开(openning)、闭合(closing)、梯度(gradient)等。

<img src="/Users/tesla/Articles/notes.md/tools/figures/j.png" />

**腐蚀(Erosion)**：基本思想类似土壤腐蚀，逐步侵蚀前景物体（白色部分）的边界，核滑过图像，仅当核下面所有的像素点都为1时原像素点的值才为1，否则被腐蚀为0。这样基于核的大小所有边界附近的像素就会被丢弃，因而就会减小前景物体，这在去除小的白噪声、分离相连物体上很有效。

```python
img = cv2.imread('j.png',0)
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)
```

<img src="/Users/tesla/Articles/notes.md/tools/figures/erosion.png" />

**膨胀(Dilation)**：与腐蚀相反，在核下面只要有元素为1像素元素就为1，这样就会增大前景物体，通常在去除噪声时，先用腐蚀去除噪声，然后用膨胀恢复物体大小，而不会恢复噪声。

```python
dilation = cv2.dilate(img,kernel,iterations = 1)
```

<img src="/Users/tesla/Articles/notes.md/tools/figures/dilation.png" />

**打开(Opening)**：腐蚀后膨胀的另一称呼，就像上面解释的一样能用于去噪：

```python
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
```

<img src="/Users/tesla/Articles/notes.md/tools/figures/opening.png" />

**闭合(Closing)**：与Opening相反，膨胀后腐蚀，能关闭前景物体上的小孔或黑点：

```python
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

<img src="/Users/tesla/Articles/notes.md/tools/figures/closing.png" />

**形态梯度(Morphological Gradients)**：图像膨胀和腐蚀之间的差值，结果就是物体的轮廓：

```python
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
```

<img src="/Users/tesla/Articles/notes.md/tools/figures/gradient.png" />

**顶帽(Top Hat)**：输入图像与打开图像间的差值，下面是$9\times9$核的结果：

```python
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
```

<img src="/Users/tesla/Articles/notes.md/tools/figures/tophat.png" />

**黑帽(Black Hat)**：输入图像与闭合图像之间的差值：

```python
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
```

<img src="/Users/tesla/Articles/notes.md/tools/figures/blackhat.png" />

**结构元素(Structuring Element)**：前面手工创建了方形的核，但有时需要椭圆/圆形核，可以使用`cv2.getStructuringElement()`函数，传递形状和核大小即可：

```python
# Rectangular Kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
# Elliptical Kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
# Cross-shaped Kernel
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
```

 

#### 三、特征检测

##### Harris角点检测

角点是图像中所有方向上密度变化都较大的区域；1988出现的Harris角点检测器寻找所有方向上位移$(u,v)$密度的差异，表达为：
$$
E(u,v) = \sum_{x,y} \underbrace{w(x,y)}_\text{window function} \, [\underbrace{I(x+u,y+v)}_\text{shifted intensity}-\underbrace{I(x,y)}_\text{intensity}]^2
$$
window函数是给出覆盖像素权值的四边形或高斯窗口。对于角点需要最大化函数$E(u,v)$，即最大化第二项，应用泰勒展开及一些数学步骤后，得到最终方程：
$$
E(u,v) \approx \begin{bmatrix} u & v \end{bmatrix} M \begin{bmatrix} u \\ v \end{bmatrix}
$$
其中
$$
M = \sum_{x,y} w(x,y) \begin{bmatrix}I_x I_x & I_x I_y \\
                                     I_x I_y & I_y I_y \end{bmatrix}
$$
而$I_x$和$I_y$则分别是$x$和$y$方向的图像导数（可使用`cv2.Sobel()`获得）。然后，创建一个确定窗口是否包含角点的打分：
$$
R = det(M) - k(trace(M))^2
$$
其中：

- $det(M) = \lambda_1 \lambda_2$
- $trace(M) = \lambda_1 + \lambda_2$
- $\lambda_1$和$\lambda_2$是$M$的特征值

这些特征值就决定了一个区域是否是角、边或平面：

- 当$\left\vert R \right\vert$很小时，这时$\lambda_1$和$\lambda_2$都很小，区域就是平坦的；
- 当$R < 0$时，必有$\lambda_1>>\lambda_2$或反之，区域就是边缘；
- 当$R$很大时，则$\lambda_1$和$\lambda_2$都很大且$\lambda_1\sim\lambda_2$，区域是角点；

如下图所示：

<img src="/Users/tesla/Articles/notes.md/tools/figures/harris_region.jpg" />

OpenCV中对应的函数为`cv2.cornerHarris()`，其参数为：

- `img`：需要是灰度图及浮点32位；
- `blockSize`：角点检测考虑的周边大小；
- `ksize`：Sobel导数的孔半径；
- `k`：方程中的Harris检测器自由函数；

```python
filename = 'chessboard.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]

cv2.imshow('dst',img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
```

有时需要最精确地找到，OpenCV提供了`cv2.cornerSubpix()`函数用亚像素正确率来进一步提炼角点。下面是一个示例：

```python
filename = 'chessboard2.jpg'
img = cv2.imread(filename)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# find Harris corners
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
dst = np.uint8(dst)

# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# Now draw them
res = np.hstack((centroids,corners))
res = np.int0(res)
img[res[:,1],res[:,0]]=[0,0,255]
img[res[:,3],res[:,2]] = [0,255,0]

cv2.imwrite('subpixel5.png',img)
```



##### Shi-Tomasi角点检测器

算法出现于1994年，将上面的打分改为$R=\min(\lambda_1,\lambda_2)$即获得。OpenCV中提供了函数`cv2.goodFeaturesToTrack()`，它使用Shi-Tomasi（也可以指定为Harris）方法找出图中$N$个最强的角点：

- 图像也必须为灰度图；
- 然后还需指明需要的角点数目；
- 然后指明质量水平，一个0-1间的值，指定角点的最小质量；

然后函数取最强的角点，丢弃其最小距离范围内的临近角点并返回$N$个最强角点。下面的示例找25个最佳角点：

```python
from matplotlib import pyplot as plt

img = cv2.imread('simple.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

plt.imshow(img),plt.show()
```



##### SIFT(Scale-Invariant Feature Transform)简介

上面的算法都是旋转不变的，但放缩后就未必。如下图：

<img src="/Users/tesla/Articles/notes.md/tools/figures/sift_scale_invariant.jpg" />

2004年不列颠-哥伦比亚大学的D.Lowe提出尺度不变特征转换算法，即SIFT，主要分为4步：

**1.尺度空间极点检测：**可以看出检测不同尺度的关键点不能使用同样的窗口，为此使用尺度空间滤波，用不同的$\sigma$值寻找Laplacian of Gaussian(LoG)。LoG充当团块检测器在不同的尺寸检测团块，而$\sigma$则是放缩参数。上面的图片中，较低值的高斯核对小角点给出很高的值，而较高值的高斯核则匹配大角点。这样就能找到不同尺度和空间的局部极大值，因而就获得$(x,y,\sigma)$的列表值。为减少消耗，SIFT使用Difference of Gaussian(DoG)来近似LoG。DoG通过一张图片两个不同$\sigma$高斯模糊的差值来获得。这个过程对高斯金字塔中图片的不同octave完成，如下所示：

<img src="/Users/tesla/Articles/notes.md/tools/figures/sift_dog.jpg" />

获得DoG后，在不同尺度和空间搜索图片的局部极值点，例如一个像素点与其周围8个相邻点、前一尺度9个像素以及后一尺度9个像素比较，若是极值点，则是潜在的关键点，它基本表示关键点能在那个尺度最好地得到表现，如下图：

<img src="/Users/tesla/Articles/notes.md/tools/figures/sift_local_extrema.jpg" />

不同参数的经验值是：octave数=4，尺度层次数=5，初始$\sigma=1.6$，$k=\sqrt2$。

**2.关键点定位：**找到潜在关键点后，还需提炼出更精确的结果。SIFT使用尺度空间的泰勒级数展开来获得极值点更精确的位置，并拒绝密度小于OpenCV中**对比阈值**（论文中偏好0.03）的极值点。DoG对边缘反馈更高，因此必须去除。为此使用一个$2\times2$的Hessian矩阵($H$)计算主曲率。在Harris角点检测器中已知，对于边缘一个特征值远大于另一个，因此丢弃比率大于OpenCV中的**边缘阈值**（论文中为10）的关键点。因此SIFT清除低对比和边缘的关键点，只留下对比度较强的兴趣点。

**3.朝向分配：**现在为每个关键点分配朝向以获得旋转不变性。基于尺度取关键点的邻域，并计算此区域的梯度大小和方向，创建一个36条纹覆盖360度的直方图，由梯度大小以及$\sigma$值等于关键点尺度1.5倍的高斯加权环形窗口来加权，取直方图的顶点，任意高于80%高峰也会考虑计算朝向，它会产生位置和尺度相同、但朝向不同的关键点。这有助于匹配的稳定性。

**4.关键点述子(descriptor)：**现在创建关键点述子。取关键点的$16\times16$邻域，将其分成16个$4\times4$子块，每个子块创建8条纹朝向直方图，因此获得共128个条纹值，表示为一个向量来形成关键点述子。此外还采取了一些方法来获得光照变化、旋转等的稳定性。

**5.关键点匹配：**通过鉴定最近邻来匹配两幅图像之间的关键点，但有些情况下第二最近匹配与第一很近，这时需拒绝近距离与第二近距离的比率的点。这会清除90%的误匹配而仅5%的正匹配。

OpenCV中提供了SIFT类：

```python
import cv2
import numpy as np

img = cv2.imread('home.jpg')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp)
cv2.imwrite('sift_keypoints.jpg',img)

img=cv2.drawKeypoints(gray,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite('sift_keypoints.jpg',img)

kp,des = sift.compute(gray,kp)

sift = cv2.SIFT()
kp, des = sift.detectAndCompute(gray,None)
```

sift.detect()函数寻找图像的关键点，若仅希望搜索图像的一部分也可以传递掩层。每个关键点都是有很多属性（坐标、邻域大小、朝向的角度、强弱的回馈等）的特殊结构。OpenCV提供了`cv2.drawKeypoints()`函数在关键点位置画小圈；若传递`cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS`标识，则会画关键点的大小和朝向。

现在要计算述子，OpenCV提供了两种方法：

- 若已有关键点，直接调用`sift.compute()`函数；
- 否则，直接用`sift.detectAndCompute()`单步找到关键点和述子。

这样就获得了关键点和描述子。



##### SURF简介

SIFT相对较慢，2006年产生了其加速版本—SURF。为寻找尺度空间，SIFT用DoG来近似LoG，而SURF则更进一步使用箱式滤波器(Box Filter)，这样的优势是以其计算的卷积可以很方便地借助积分图像计算。并且不同尺度可以并行实现。对于尺度和位置，SURF同样依赖于Hessian的行列式。下图是这种近似的一个展示：

<img src="/Users/tesla/Articles/notes.md/tools/figures/surf_boxfilter.jpg" />

对于朝向，SURF使用周围6s邻域水平和垂直方向的小波反馈和适当的高斯权值。通过计算一个60度角滑动窗口内所有反馈的和来评估主要朝向（如下图）。有趣的是使用积分图像能非常方便地在任意尺度找出小波回馈。许多应用并不要求旋转不变性，可以不计算朝向，这样能加速这个过程。SURF提供一种Upright-SURF或U-SURF功能，能改善速度以及达$\pm 15^{\circ}$的稳定性。OpenCV提供了`upright`标识，若为0，则计算朝向；为1，则不。

<img src="/Users/tesla/Articles/notes.md/tools/figures/surf_orientation.jpg" />

对特征描述，SURF使用关键点周围$20s\times20s$领域水平和垂直方向的小波反馈。



#### 四、视频分析

##### 1.Meanshift

Meanshift背后的直觉是，考虑一系列点（可以是类似直方图反向投影的像素分布），给定一个窗口（比如一个圆），需要将其移动到像素密度最大的区域。如下图所示：

<img src="/Users/tesla/Articles/notes.md/tools/figures/meanshift_basics.jpg" />

初始的窗口圆为蓝色，其中心为C1\_o，质心为C1\_r；移动圆使其中心为原来的质心C1\_r；重复这个过程直到圆心与质心重合，最终获得绿色的圆。下面的图就展示了这个过程：

<img src="/Users/tesla/Articles/notes.md/tools/figures/meanshift_face.gif" />

因此一般传递直方图反向投影的图像和初始目标位置，显然当目标移动时，这种移动会反映到直方图反向投影，这样meanshift算法会将移动到新的密度最大的地方。

要在OpenCV中使用meanshift算法，首先需要设定目标，找到其直方图，这样才能将每一帧的目标反向投影用于计算；另外也要提供初始窗口。这里仅考虑Hue直方图，而为了避免低光照导致的错误，也会使用`cv2.inRange()`将低光照丢弃。

```python
import numpy as np
import cv2

cap = cv2.VideoCapture('slow.flv')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()
```



##### 2.Camshift

Meanshift的一个问题是当物体移动时，窗口大小一直不变，这样当物体相对摄像机的远近发生变化时窗口就会显得太小或太大。这就需要根据目标的旋转和大小调整窗口大小。1988年Gary Bradsky提出的CAMshift(Continuously Adaptive Meanshift)就用于解决这个问题。它首先应用meanshift，当其收敛后便更新窗口大小为$s=2\times\sqrt{\frac{M_{00}}{256}}$。它也计算最匹配椭圆的朝向，并再次用放缩后的窗口和前一窗口位置应用meanshift。这个过程持续一直到满足准确率：

<img src="/Users/tesla/Articles/notes.md/tools/figures/camshift_face.gif" />

OpenCV中camshift用法几乎与meanshift一样，但返回一个旋转的四边形（这就是结果）和箱式的参数（用于下次迭代的搜索窗口）：

```python
import numpy as np
import cv2

cap = cv2.VideoCapture('slow.flv')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()
```



##### 3.光流

光流是连续两帧之间由物体或摄像机移动引起的图像物体明显移动的模式，是一个2D向量场，每个向量都展示点移动的偏移，比如下图：

<img src="/Users/tesla/Articles/notes.md/tools/figures/optical_flow_basic1.jpg" />

考虑第一帧的一个像素$I(x,y,t)$（注意这里增加了时间$t$这个维度），时间$dt$后的下一帧移动了距离$(dx,dy)$。因为这些像素相同且密度不变，可以说：
$$
I(x,y,t) = I(x+dx, y+dy, t+dt)
$$
即$I(x,y,t)$为确定像素像素点的函数，$ I(x+dx, y+dy, t+dt)$与其虽然自变量的值，即坐标和时间，不同，但都能确定是同一个像素点。取等式右边的泰勒级数，去除共同项并除以$dt$得到：
$$
f_xu + f_yv + f_t = 0
$$
其中
$$
f_x = \frac{\partial f}{\partial x} \; ; \; f_y = \frac{\partial f}{\partial x}\\
u = \frac{dx}{dt} \; ; \; v = \frac{dy}{dt}
$$
上式为光流方程，可以发现$f_x$和$f_y$是图像梯度，$f_t$是时间梯度，但$(u,v)$未知，无法用一个方程解决两个未知数。Lucas-Kanade是一种解决方法。可以假设所有i相邻像素的移动相同，Lucas-Kanade取点周围$3\times3$块，因此9个点的运动相同，可以找到这9个点的$(f_x,f_y,f_t)$，因此就变成9个方程2个未知数，使用最小二乘拟合方法，下面是最终的结果：
$$
\begin{bmatrix} u \\ v \end{bmatrix} =
\begin{bmatrix}
    \sum_{i}{f_{x_i}}^2  &  \sum_{i}{f_{x_i} f_{y_i} } \\
    \sum_{i}{f_{x_i} f_{y_i}} & \sum_{i}{f_{y_i}}^2
\end{bmatrix}^{-1}
\begin{bmatrix}
    - \sum_{i}{f_{x_i} f_{t_i}} \\
    - \sum_{i}{f_{y_i} f_{t_i}}
\end{bmatrix}
$$
用Harris角点检测器验证逆矩阵的相似性，它表示角点是更好的追踪点。这些解决的是小移动，当移动很大时，就再次要用到金字塔。当沿着金字塔向上时，小移动就会被去除而大移动就会变成小移动。因此应用LIcas-Kadane，得到带尺度的光流。

OpenCV用单个函数`cv2.calcOpticalFlowPyrLK()`提供了所有功能。这里，使用`cv2.goodFeaturesToTrack`来决定追踪点，检测第一帧的与一些Shi-Tomasi角点，然后用Lucas-Kanade光流迭代地追踪这些点。将前一帧、前些点和下一帧传递给函数`cv2.calcOpticalFlowPyrLK()`，它返回后些点和状态数，若1表示找到下一点，否则0。如此持续迭代，代码如下：

```python
import numpy as np
import cv2

cap = cv2.VideoCapture('slow.flv')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
```

这个代码并未检查下个关键点的正确性，即便有些特征点在图像中消失，依然有可能找到其下一个点。因此需要按一定的时间间隔检测角点，可以是每5帧等。下面展示了一个结果：

<img src="/Users/tesla/Articles/notes.md/tools/figures/opticalflow_lk.jpg" />



##### 4.密集光流

Lucas-Kanade方法计算稀疏特征集的光流，比如上面的例子中是角点。OpenCV中提供了另一个计算密集光流的算法，基于2003年Gunner Farneback提出的算法，为一帧的几乎所有点计算光流。下面的示例展示了如何使用上面算法找到密集光流，获得2-通道的光流向量$(u,v)$，找到其大小和方向，方向对应于色调值，大小则对应于平面值。

```python
import cv2
import numpy as np
cap = cv2.VideoCapture("vtest.avi")

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()
```

下面是结果：

<img src="/Users/tesla/Articles/notes.md/tools/figures/opticalfb.jpg" />



##### 5.背景消除

若有单独的背景图像，则将新图像减去背景就能获得前景物体。但大多数时候并没有这样的图像，因此就必须从拥有的无论如何的图像中提取出背景。当有阴影时会更复杂，因为它也会移动。简单的相减也会将其标记为前景。OpenCV实现了三个用于此的算法。

**BackgroundSubtractorMOG**：这是一个基于高斯混合的的背景/前景分割算法，由P. KadewTraKuPong和R. Bowden在2001年的论文《An improved adaptive background mixture model for real-time tracking with shadow detection》中提出。它通过$K$($K=3\sim5$)个高斯分布建立每个背景像素的模型，混合参数表示这些颜色停留在屏幕上的时间比例。最可能的背景色就是在屏幕上停留得更长更固定的颜色。当编程时，需要使用函数`cv2.createBackgroundSubtractorMOG()`创建背景对象，然后在视频循环中使用`backgroundsubtractor.apply()`来获得前景掩层：

```python
import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

fgbg = cv2.createBackgroundSubtractorMOG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

**BackgroundSubtractorMOG2**：也是基于高斯混合的背景/前景分割算法，它的一个重要特征是会为每个像素选择合适的高斯分布数$K$，基于光照变化为不同场景提供了更好的可调节性。编程时也需要创建一个背景提取器对象，这里可以选择是否检测阴影，若`detectShadows=True`（默认设定），检测并标记阴影。阴影会被标记为灰色：

```python
import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

**BackgroundSubtractorGMG**：这个算法结合了统计背景图像估计和贝叶斯分割。它使用前面一些帧（默认120）来建立背景模型，部署用贝叶斯推断来辨别可能的前景物体的概率前景分割算法；估计是自适应的，新的观察会被赋予更多的权值以适应变化的照明。应用了几种形态过滤操作来去除不需要的噪声。

```python
import numpy as np
import cv2

cap = cv2.VideoCapture('vtest.avi')

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
fgbg = cv2.createBackgroundSubtractorGMG()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

