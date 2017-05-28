# emotion-analysis
This is a project for running an emotional analysis test on a face to predict the emotion portrayed by the person and play an appropriate song.
The project is implemented using python; Dlib and scikit learn libraries are the main libraries used here. 
This application can detect for the following emotions:
  1. Happy
  2. Sad
  3. Neutral
  4. Fear
  5. Disgust
  6. Surprise
  7. Contempt
  8. Anger
  
I use facial landmarks for detecting and extracting the features from the face and then feed them into an SVM. For real-time analysis, the application will consider the emotion portrayed in first 10 frames and then will play a song depending on the emotion detected.
The songs to be played for each emotion will be stored in an excel sheet and played by the application. Before testing the classifier, I have trained it on CK+ dataset and a home brewed dataset consisting of images of me and my classmates showing different emotions.
The model can be re-trained as per convenience.
The following list explains functionality of each file:
  1. orderingDataset.py    -Module for ordering the CK+ dataset.
  2. faceDetectionDlib3.py -File consists modules for facial landmark detection and emotion calssification.(program to be run for training and static image testing)
  3. webcamFeed3.py        -Module for running the webcam.(program to be run for real-time emotion analysis)
  4. musicPlayer.py        -Module for playing song according to the emotion detected.
  5. emo_analysis.pkl      -Trained SVM for emotion analysis
  6. EmotionLinks.xlsx     -Excel sheet consisting paths to songs to be played for each emotion.
  
On training with CK+ dataset I obtained a testing accuracy of 78%. When trained with CK+ and home-brewed dataset I obtained a testing accuracy of 84%.



