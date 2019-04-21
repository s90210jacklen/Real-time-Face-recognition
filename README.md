# Real-time Face recognition 即時人臉辨識 </br> (Using Keras and Tensorflow)
分別來自《DeepFace: Closing the gap to human-level performance in face verification》(2014)與《FaceNet: A Unified Embedding for Face Recognition and Clustering》(2015)這兩篇paper提出的方法，而外利用OpenCV來擷取Webcam影像並使用其提供的Haar Cascade進行人臉檢測(Face Detection)

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
  - if d(img1, img2) ≦  τ　→  **Same**
  - if d(img1, img2) >  τ　→   **Different**</br>
  
  如此一來就解決了Face Verification (人臉驗證)1:1 matching的問題
  
![one shot](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/one-shot_learning_1.png)

![one shot2](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/one-shot_learning_2.png)

- **Siamese network (孿生網路)**</br>

  - 使用Siamese network的架構來達成上述Similarity Function的效果，其實就是使用兩個常見的ConvNet的網路架構，這個兩個網路擁有相同的參數與權重，一樣經由Convolution(卷積)、Pooling(池化)、Fully connected layers(全連接層)最後得到一個帶有128個數字的特徵向量(feature vector)，而這個過程稱為encoding(編碼)
  
![arch](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/arch.png)
  - 將兩張圖片(這裡稱x(1)與x(2))放入這兩個ConvNet後得出編碼後的兩個特徵向量(feature vector)
  - 為了算出兩張圖片相似度，方式為將這兩個經由編碼所獲得的128維特徵向量f(x1)、f(x2)相減並取2範數(2-Norm)，這樣我們就透過Siamese network學習出我們所想要的Similarity Function(相似度函數)
  ![different](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/different.png)
  
**Note**: 2範數(2-Norm)又稱為為歐基里德範數(Euclidean norm)，是以歐式距離的方式作為基礎，計算出向量的長度或大小
![L2_distance](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/L2%20distance.png)

  - 總結來說，在Siamese network的架構我們希望能學出一種encoding(編碼)方式，更準確來說是希望學習出參數使得我們能達成以下的目標
  ![define encoding](https://github.com/s90210jacklen/Real-time-Face-recognition/blob/master/images/Define_decoding.png)
 在上述的目標中，改變ConvNet每一層的參數就會得到不同的編碼，所以我們可以利用反向傳播(Backpropagation)來更改這些參數以達到上列的目標



- **Triplet Loss (三元組損失)**</br>
在NN(Neural Network)的訓練中，都需要一個損失函數(Loss function)作為最小化(minimize)目標，而在Face recognition的應用中為了能夠學習參數來得到良好的encoding(編碼)，《FaceNet: A Unified Embedding for Face Recognition and Clustering》這篇論文提出一種有效的損失函數稱為**Triplet Loss (三元組損失)**

在Triplet Loss中我們都需要比較兩隊
