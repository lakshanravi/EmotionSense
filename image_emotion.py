import cv2                 # Import OpenCV library
import numpy as np      # Import numpy library for numerical computations
from keras.models import load_model

# Load pre-trained model
model=load_model('model_file.h5')


# Load Haar Cascade Classifier for face detection
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

# len(number_of_image), image_height, image_width, channel

# Read and process image
frame=cv2.imread("test03.jpg")      # Read an image from file
gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)            # Convert the image to grayscale
faces= faceDetect.detectMultiScale(gray, 1.3, 3)        # Detect faces in the grayscale image

# Detect faces and predict emotions
for x,y,w,h in faces:
    sub_face_img=gray[y:y+h, x:x+w]
    resized=cv2.resize(sub_face_img,(48,48))
    normalize=resized/255.0
    reshaped=np.reshape(normalize, (1, 48, 48, 1))
    
    # Use the loaded model to predict the emotion label for each face. train krpu model ek ntm use krnne mekt
    result=model.predict(reshaped)
    label=np.argmax(result, axis=1)[0]   # Get the index of the maximum value (predicted emotion)
    print(label)  #print lable
    
    
    # Draw rectangles around the detected faces
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
    
    # Draw rectangle for displaying label
    cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
    
    # Display the predicted emotion label on the image

    cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    
# Display results    
        
cv2.imshow("Frame",frame)   #Display the processed image with face detection and emotion labels

cv2.waitKey(0)
cv2.destroyAllWindows()