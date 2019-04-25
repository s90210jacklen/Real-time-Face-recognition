# Real-time Face recognition 即時人臉辨識 </br> (Using Keras, Tensorflow and OpenCV)
分別來自[《DeepFace: Closing the gap to human-level performance in face verification》(2014)](#Reference)[1]與[《FaceNet: A Unified Embedding for Face Recognition and Clustering》(2015)](#Reference)[2]這兩篇paper提出的方法，而外利用OpenCV來擷取Webcam影像並使用其提供的Haar Cascade分類器進行人臉檢測(Face Detection)

在Face Recognition(人臉辨識)的問題上，通常會再進一步分成兩個種類 :
- **Face Verification (人臉驗證) :** 
  - 給予輸入image, name/ID
  - 輸出是否為此人
  - 視為 1:1 matching problem
  
  e.g. 手機的人臉解鎖

- **Face Recognition (人臉辨識) :** 
  - 擁有K個人物的Database
  - 給予Input image
  - 輸出ID, if (image為K個人物中的其中一個) </br>
  無法辨識此人, if (image不為K個人物中任何一個)
  - 視為 1:K matching problem
  
  e.g. 使用的人臉辨識的員工通行閘門
   
# Concept
在Face Recognition(人臉辨識)的應用中經常要做到只靠一張照片就能辨認一個人，但深度學習(Deep Learning)的演算法在只有一筆訓練資料的情況下效果會很差，所以在人臉辨識中必須解決**One Shot Learning**(單樣本學習)的問題

- **One Shot Learning (單樣本學習)**</br>

假定某公司內的Database共有4位人員的照片各一張，當有其中一位人員經過系統前的鏡頭並被捕捉到臉孔後，儘管Database只有一張此人的照片，系統依然能辨認出此臉孔為公司裡的員工，相反的，若不為公司內人員則無法辨識此人


- **Similarity Function (相似度函數)**</br>
為了達到One Shot Learning (單樣本學習)這樣的目標，我們希望讓NN(Neural Network)去學習一個函數**d**</br></br>
**d(img1, img2)** : 給予兩張照片，輸出這兩張照片的相異程度
  - 如果兩張照片是同一個人，則輸出一個較小的數字
  - 如果兩張照片是不同人，則輸出一個較大的數字</br>
**此外，需定義一Hyperparameter(超參數)「τ」**
  - if d(img1, img2) ≤  τ　→  **Same**
  - if d(img1, img2) >  τ　→   **Different**</br>
  
  如此一來就解決了Face Verification (人臉驗證)1:1 matching的問題
  
![one shot](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/one-shot_learning_1.png)

![one shot2](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/one-shot_learning_2.png)

- **Siamese network (孿生網路)**</br>

  - 使用[《DeepFace: Closing the gap to human-level performance in face verification》(2014)](#Reference)[1]提出的Siamese network架構來達成上述Similarity Function的效果，其實就是使用兩個常見的ConvNet的網路架構，這個兩個網路擁有相同的參數與權重，一樣經由Convolution(卷積)、Pooling(池化)、Fully connected layers(全連接層)最後得到一個帶有128個數字的特徵向量(feature vector)，而這個過程稱為encoding(編碼)
  
![arch](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/arch.png)
   - 將兩張圖片(這裡稱x(1)與x(2))放入這兩個ConvNet後得出編碼後的兩個特徵向量(feature vector)
   - 為了算出兩張圖片相似度，方式為將這兩個經由編碼所獲得的128維特徵向量f(x1)、f(x2)相減並取2範數(2-Norm)的平方，這樣我們就透過Siamese network學習出我們所想要的Similarity Function(相似度函數)
  ![different](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/different.png)
  
**Note:** 2範數(2-Norm)又稱為為歐基里德範數(Euclidean norm)，是以歐式距離的方式作為基礎，計算出向量的長度或大小
![L2_distance](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/L2%20distance.png)

  - 總結來說，在Siamese network的架構我們希望能學出一種encoding(編碼)方式，更準確來說是希望學習出參數使得我們能達成以下的目標
  ![define encoding](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/Define_decoding.png)
 在上述的目標中，改變ConvNet每一層的參數就會得到不同的編碼，所以我們可以利用反向傳播(Backpropagation)來更改這些參數以達到上列的目標



- **Triplet Loss (三元組損失)**</br>
在NN(Neural Network)的訓練中，都需要一個損失函數(Loss function)作為最小化(minimize)目標，而在Face recognition的應用中為了能夠學習參數來得到良好的encoding(編碼)，[《FaceNet: A Unified Embedding for Face Recognition and Clustering》(2015)](#Reference)[2]這篇論文提出一種有效的損失函數稱為**Triplet Loss (三元組損失)**

    - 在Triplet Loss中會有Anchor、Positive、Negative這三種照片
    - Positive為與Anchor**同個人**的照片
    - Negative則為**不同人**的照片
    - 我們需要比較Anchor分別與Positive和Negative一組的兩對的照片
    - 目標是希望Anchor與Positive的距離(編碼)較近，與Negative的距離(編碼)較遠
  
也就是說，我們希望神經網路的參數所造成的編碼能夠使Anchor與Positive的距離**小於等於**Anchor與Negative的距離這樣的性質</br>

![Learning objective](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/Learning%20objective.png)

- 在上圖中，Anchor、Positive、Negative分別簡寫為A、P、N
  - 如果f變成**零函數**會將每個向量的輸出都變成零，就是所謂的**trivial solutions**，則0 - 0 ≤ 0 這樣就很容易滿足這個不等式，會讓NN學不到我們的目標
  - 為了不讓NN將編碼學習成零函數，我們希望兩對的照片的差距不只小於等於零，還要**比零還小一些**，因而外引進一個超參數(Hyperparameter)**α**，這個α稱為margin(邊距)，我們讓≤這個符號左邊的式子小於負α，習慣上會將α移到式子左邊
  - 而margin(邊距)用意即是拉開d(A,P)與d(A,N)這兩對的差距，就是把這兩對推開，**盡量的遠離彼此**</br>
  e.g. 假設margin = 0.2 ,表示若d(A,P)=0.5 則d(A,N)至少0.7才符合上述的不等式，若d(A,N)為0.6就不符合，因為兩組的差距不夠大
  - Triplet Loss的實現如下
  ```python
  def triplet_loss(y_true, y_pred, alpha = 0.3):
  
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    計算anchor和positive的編碼(距離)
    # Step 1: 計算anchor和positive的編碼(距離)
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: 計算anchor和negative的編碼(距離)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: 將先前計算出的距離相減並加上邊距alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: 將上述計算出的損失與零取最大值，再將所有樣本加總起來
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss
  ```
  
- **Loss Function (損失函數)**</br>
Triplet Loss定義在3張一組的圖片A、P、N上，則損失函數則可以定義成:
![Loss Function](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/total%20cost.png)</br>

這個max函數的用意在於，若括號的左邊項 ≤ 0則損失就為零，若左邊項 > 0則損失變成>零；而我們是希望損失越小越好，所以只要左邊項≤ 0不管負多少，就能把損失推向零</br>

- **Cost function (成本函數)**</br>
將訓練資料裡一組三張圖片的損失加總起來作為整體NN的總成本(Total cost)，並利用Gradient descent(梯度下降法)來去訓練NN最小化成本

![cost Function](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/cost_function.png)

**Note:** 假定有10000張訓練圖片，分別來自1000個不同的人(每人約10張圖片)才能構成我們的資料集，若每個人只有一張照片這樣就無法順利挑出Anchor與Positive，但是當NN訓練完成後就可以將系統用在One-shot Learning的問題，對於你想辨識的人，你可能只有他的一張照片也能順利辨識出此人。

- **Choosing the triplets A, P, N**</br>
  - 在訓練資料中，Triplets(三元組)樣本的選擇會是一個問題，因為在上述學習目標 **d(A,P) + α ≤ d(A,N)** 中，若只按照要求隨機的選擇同一個人的照片A與P
和不同人照片A與N，則這個不等式很容易就被滿足，因為隨機挑兩個人的照片有很大的機率使得A與N差異遠大於A與P，這會使得NN無法學習有效的參數

  - 因此，要建立訓練集的話必須挑選那種很難訓練的A,P和N，因為目標是讓所有Triplets(三元組)滿足**d(A,P) + α ≤ d(A,N)** 這個不等式，而很難訓練的Triplets(三元組)的意思就是你所挑選的A,P和N會讓 **d(A,P)≈ d(A,N)** ，如此一來NN在學習的時候就必須花更大的力氣嘗試讓**d(A,N)往上推**或讓**d(A,P)往下掉** ，推開彼此以達到相隔α的邊距，這樣的效果會讓你的學習演算法更效率；反之，若隨便選會導致很多的Triplets(三元組)都解起來很簡單，Gradient descent(梯度下降法)就不會再做任何事，因為你的NN早已把問題都做對了，在這部分在[《FaceNet: A Unified Embedding for Face Recognition and Clustering》(2015)](#Reference)[2]這篇論文有更詳細的說明
  
- **Face detection (人臉偵測)**</br>
在人臉偵測的部分使用OpenCV的Haar Cascade分類器，選擇的為人臉分類器[haarcascade_frontalface_default.xml](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/haarcascade_frontalface_default.xml)

# Usage

**On Windows**
```bash
eg: With Tensorflow as backend
> python facenet.py 
```

# Result
- 利用OpenCV的Haar Cascade分類器進行人臉偵測(Face detection)</br></br>
![Output](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/output.png)

- 偵測出人臉後使用預訓練的FaceNet來進行encoding並計算距離，辨識從Webcam讀取的影像是否為資料庫中的人物
  - 若距離**小於0.7**則回傳資料庫內對應的人名與印出字串並發出語音"Welcome (someone), have a nice day!"
  - 若不是則繼續偵測人臉並計算當下影像的編碼與距離
![output distance](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/output%20distance.png)  

# References
- [1] [《DeepFace: Closing the gap to human-level performance in face verification》](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)
- [2][《FaceNet: A Unified Embedding for Face Recognition and Clustering》](https://arxiv.org/pdf/1503.03832.pdf)
- [3][Face Detection in Python Using a Webcam](https://realpython.com/face-detection-in-python-using-a-webcam/)
- [4][OpenFace pretrainde model](https://github.com/iwantooxxoox/Keras-OpenFace)
- [5][Official FaceNet github repository](https://github.com/davidsandberg/facenet)
