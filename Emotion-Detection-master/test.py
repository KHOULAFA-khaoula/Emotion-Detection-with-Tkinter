



#USAGE : python test.py


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np


face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Furieux','Heureux','Neutre','Triste','Surpris']


cap = cv2.VideoCapture(0)



def detection():
        
      
      


     #   while True:

            # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                # make a prediction on the ROI, then lookup the class

                    preds = classifier.predict(roi)[0]
                    highest_prediction=max(preds)
                    print("\nprediction = ",preds)
                    label=class_labels[preds.argmax()]
                    result=label+" "+str(round(highest_prediction,4)*100)+"%"
                    print("\nprediction max = ",preds.argmax())
                    print("\nlabel = ",label)
                    label_position = (x,y)
                    cv2.putText(frame,result,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),3)
                else:
                    cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                print("\n\n")
        cv2.imshow('Emotion Detector',frame)
          #  if cv2.waitKey(1) & 0xFF == ord('q'):
               # break
         
        root.after(1,detection)


from tkinter import *
from PIL import ImageTk,Image


def stop_detection():
    cap.release()
    cv2.destroyAllWindows()
   
def quit():
    root.destroy()

root = Tk()


root.title("Emotion Detection Application")
root.geometry("850x650")
root.configure(bg='#9cbab2')


class BackgroundImage(Frame):
    def __init__(self, master, *pargs):
        Frame.__init__(self, master, *pargs)


        
        self.image = Image.open(".\\background2.jpg")
        self.img_copy= self.image.copy()


        self.background_image = ImageTk.PhotoImage(self.image)

        self.background = Label(self, image=self.background_image)
        self.background.pack(fill=BOTH, expand=YES)
        self.background.bind('<Configure>', self._resize_image)

        self.stop_detection_button =  Button(root ,
                                             text ="StopDetection",
                                            command=stop_detection ,
                                             bg="#fff",
                                             justify=LEFT)
       
        self.quit_button =  Button(root, 
                                    text ="Quit Appilcation", 
                                    command=quit,
                                    bg="#fff",
                                    justify=RIGHT)

        
        self.start_detection_button =  Button(root,
                                               text="Start Detection", 
                                               command=detection ,
                                               bg="#fff",
                                             justify=CENTER)

       
        self.quit_button.pack()
        self.start_detection_button.pack()
        self.stop_detection_button.pack()

    def _resize_image(self,event):

        new_width = event.width
        new_height = event.height

        self.image = self.img_copy.resize((new_width, new_height))

        self.background_image = ImageTk.PhotoImage(self.image)
        self.background.configure(image =  self.background_image)


e = BackgroundImage(root)
e.pack(fill=BOTH, expand=YES)



#buuttons#


#Display Buttons 
#button1_canvas = canvas1.create_window( 100, 10, 
#                                        anchor = "nw",
#                                        window = stop_detection_button)
  
# button2_canvas = canvas1.create_window( 100, 40,
#                                        anchor = "nw",
#                                        window = quit_button)
  
# button3_canvas = canvas1.create_window( 100, 70, anchor = "nw",
#                                        window =start_detection_button)

# canvas1.pack(fill = "both", expand = True)




root.mainloop()






























