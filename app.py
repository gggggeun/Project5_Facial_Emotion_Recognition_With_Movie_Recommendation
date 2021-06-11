import numpy as np
import pandas as pd
import cv2
import streamlit as st
from model_1 import model_1
from selenium import webdriver
import random



# facial emotion recognition
show_text=[0]
emotion_dict = {0: "   Angry  ", 1: "Bad", 2: "  Fearful  ", 3: "   Happy   ", 4: "  Bored  ", 5: "    Sad    ", 6: "Surprised"}
st.title("Webcam Live Feed")
run = st.button('Run')

FRAME_WINDOW = st.image([])
FRAME_WINDOW2 = st.image([])
FRAME_WINDOW3 = st.image([])
recommend = st.button('Recommend')
cap = cv2.VideoCapture(0)
ret, frame = cap.read()



# press run
while run:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    # face detection
    frame = cv2.resize(frame,(600,500))
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
    # FRAME_WINDOW2.image(gray_frame)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 255, 255), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        # FRAME_WINDOW.image(frame)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = model_1.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        predicted = emotion_dict[maxindex]
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
        FRAME_WINDOW2.image(frame)


# press recommend
if recommend:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
    # face detection
    frame = cv2.resize(frame,(600,500))
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
    # FRAME_WINDOW2.image(gray_frame)
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 255, 255), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        # FRAME_WINDOW.image(frame)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = model_1.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        predicted = emotion_dict[maxindex]
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
        FRAME_WINDOW2.image(frame)

        st.text("You look....")
        st.write(predicted)
        st.text("I'd like to recommend a movie to you called...")
    emotion = predicted    
    if emotion == 'Happy':
        #genre:Adventure
        url ='https://www.imdb.com/search/title/?genres=adventure&explore=title_type,genres&view=advanced'
    elif emotion == "Angry":
        #genre:Crime
        url ='https://www.imdb.com/search/title/?genres=crime&explore=title_type,genres&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=f1cf7b98-03fb-4a83-95f3-d833fdba0471&pf_rd_r=T5MTHH7K7WKVACTFFSJW&pf_rd_s=center-3&pf_rd_t=15051&pf_rd_i=genre&ref_=ft_gnr_pr3_i_3' 
    elif emotion == "Bad":
        #genre:comedy
        url ='https://www.imdb.com/search/title/?genres=comedy,romance&explore=title_type,genres&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=a581b14c-5a82-4e29-9cf8-54f909ced9e1&pf_rd_r=T5MTHH7K7WKVACTFFSJW&pf_rd_s=center-5&pf_rd_t=15051&pf_rd_i=genre&ref_=ft_gnr_pr5_i_1'
    elif emotion == "Fearful":
        #genre:superhero
        url ='https://www.imdb.com/search/keyword/?keywords=superhero&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=a581b14c-5a82-4e29-9cf8-54f909ced9e1&pf_rd_r=T5MTHH7K7WKVACTFFSJW&pf_rd_s=center-5&pf_rd_t=15051&pf_rd_i=genre&ref_=ft_gnr_pr5_i_3'
    elif emotion == 'Bored':
        #genre:action
        url ='https://www.imdb.com/search/title/?genres=action,comedy&explore=title_type,genres&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=a581b14c-5a82-4e29-9cf8-54f909ced9e1&pf_rd_r=T5MTHH7K7WKVACTFFSJW&pf_rd_s=center-5&pf_rd_t=15051&pf_rd_i=genre&ref_=ft_gnr_pr5_i_2'
    elif emotion == 'Sad':
        #genre:drama
        url ='https://www.imdb.com/search/title/?genres=drama&explore=title_type,genres&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=f1cf7b98-03fb-4a83-95f3-d833fdba0471&pf_rd_r=T5MTHH7K7WKVACTFFSJW&pf_rd_s=center-3&pf_rd_t=15051&pf_rd_i=genre&ref_=ft_gnr_pr3_i_1'
    else: #emotion == 'Surprised' genre:fantasy
        url ='https://www.imdb.com/search/title/?genres=fantasy&explore=title_type,genres&pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=fd0c0dd4-de47-4168-baa8-239e02fd9ee7&pf_rd_r=T5MTHH7K7WKVACTFFSJW&pf_rd_s=center-4&pf_rd_t=15051&pf_rd_i=genre&ref_=ft_gnr_pr4_i_3'
    def movie_recommend():
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument("start-maximized")
        options.add_argument("disable-infobars")
        options.add_argument('window-size=1920x1080') 
        options.add_argument("disable-gpu")
        options.add_argument("--disable-extensions")

        
        driver = webdriver.Chrome(r'C:\Users\geun\chromedriver.exe', chrome_options=options)
        driver.get(url)

        #Set initial empty list for each element:
        title = []
        #Grab the block of each individual movie
        block = driver.find_elements_by_class_name('lister-item')
        #Set up for loop to run through all 50 movies
        for i in range(0,50):
            #Extracting title
            ftitle = block[i].find_element_by_class_name('lister-item-header').text
            title.append(ftitle)
        return random.choice(title)

    st.write(movie_recommend())