import cv2

face_cap = cv2.CascadeClassifier("C:/Users/benss/Desktop/Bens/opencv app/myenv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml") #enter your file path in between the quotes and replace "\" to "/"


#The below code for live video
video_cap=cv2.VideoCapture(0)#to start pc camera on runtime and capture live pictures
while True:
    ret,video_data = video_cap.read() #ret and video are variables and read() is used to read the file
    col=cv2.cvtColor(video_data,cv2.COLOR_BGR2GRAY) #converting the images to b&w
    faces = face_cap.detectMultiScale(          #recolorizing
        col,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in faces:
        cv2.rectangle(video_data,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("video_live",video_data) #to obta(in the live video frame
    if cv2.waitKey(10)==ord("a"):       #to stop the loop and video press 'a' key in your keyboard
        break
video_cap.release()    
