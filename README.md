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
使用Siamese network的架構來達成上述Similarity Function的效果，其實就是使用兩個常見的ConvNet的網路架構，一樣經由Convolution(卷積)、Pooling(池化)、Fully connected layers(全連接層)最後得到一個帶有128個數字的特徵向量(feature vector)
