
# 🎭 Emotion Detection with Deep Learning and OpenCV

This AI-powered project detects human emotions from images and videos using a Convolutional Neural Network (CNN) and Haar Cascade face detection. It's ideal for real-time applications such as emotion-aware systems, surveillance, education, and more.


## 📌 Features

- 🤖 Emotion recognition from faces in images and videos
- 📸 Face detection using Haar Cascade Classifier
- 🧠 Trained CNN model (Keras/TensorFlow)
- 🖼️ Image mode and 🎥 video mode support
- 🔤 Emotions detected: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise



## 🧠 Emotion Classes

0     Angry    
1     Disgust   
2     Fear      
3     Happy     
4     Neutral   
5     Sad       
6     Surprise 



## 🚀 How to Run

### 📷 Detect Emotions in an Image

```bash
python image_emotion.py
```

This reads an image (e.g., `test03.jpg`), detects faces, and classifies the emotion.



### 🎞️ Detect Emotions in a Video

```bash
python video_emotion.py
```

This loads a video (e.g., `sample_video.mp4`), detects faces frame-by-frame, and overlays emotion predictions.



