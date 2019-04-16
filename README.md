# Real-time Face recognition 即時人臉辨識 (Using Keras and Tensorflow)
分別來自《DeepFace: Closing the gap to human-level performance in face verification》(2014)與《FaceNet: A Unified Embedding for Face Recognition and Clustering》(2015)這兩篇paper中提出的方法，而外利用OpenCV來擷取Webcam影像並使用其提供的Haar Cascade進行人臉檢測(Face Detection)

在Face Recognition(人臉辨識)的問題上，通常會再進一步分成兩個種類 :
- Face Verification (人臉驗證) : 
  - 給予輸入image, name/ID
  - 輸出是否為此人
  - 視為 1:1 matching problem
  
  eg: 手機的人臉解鎖

- Face Recognition (人臉辨識) : 
  - 擁有K個人物的Database
  - 給予Input image
  - 輸出ID, if (image為K個人物中的其中一個) </br>
  無法辨識此人, if (image不為K個人物中任何一個)
  - 視為 1:K matching problem
  
  eg: 某些公司所使用的人臉辨識員工通行閘門
   
