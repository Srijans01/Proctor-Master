import sys
import os
import datetime
import pyttsx3 
import speech_recognition as sr 
from tkinter import *

engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')

def speak(audio):
    engine.say(audio)
    engine.runAndWait()

window=Tk()

window.title("Proctor Master")
window.geometry('550x200')
f1= Frame(window,bg="yellow",borderwidth=7,relief="sunken").pack(side="left",fill="y")
C=Canvas(window,bg="blue",height=250,width=300)
fn=PhotoImage(file="backp.png")
background_label=Label(window,image=fn)
background_label.place(x=0,y=0,relwidth=1,relheight=1)
C.pack()
p1 = PhotoImage(file = 'logo_1.png')
window.iconphoto(False, p1) 


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=0 and hour<12:
        speak("Good Morning!")

    elif hour>=12 and hour<18:
        speak("Good Afternoon!")   

    else:
        speak("Good Evening!")  

    speak("Your exam will start in  a moment,please wait patiently")

def exitt():
    window.destroy()

def inst_win():
    top8 = Tk()
    top8.title("Instructions")
    top8.geometry('500x200')
    l27=Label(top8,text="1.To start the exam click on START")
    l27.pack()
    l28=Label(top8,text="2.The examinee will be given warnings if found to be invloved in any type of malpractices.")
    l28.pack()
    l29=Label(top8,text="3.After 5 warnings , the exam will automatically be cancelled")
    l29.pack()
    l30=Label(top8,text="4.Always ensure not looking away from camera for prolonged period of time and don't use mobile phone while giving exam.")
    l30.pack()

def instruct():
    inst_win()
    speak("To start the exam click on START")
    speak("The examinee will be given warnings if found to be invloved in any type of malpractices.")
    speak("After 5 warnings , the exam will automatically be cancelled") 
    speak("Always ensure not looking away from camera for prolonged period of time and don't use mobile phone while giving exam.")           


def run():
    wishMe()
    os.system('head_pose_estimation.py')

def run1():
    os.system('person_and_phone.py')


f1= Frame(window,bg="yellow",borderwidth=7,relief="sunken").pack(side="left",fill="y")
C=Canvas(window,bg="blue",height=250,width=300)
fn=PhotoImage(file="backp.png")
background_label=Label(window,image=fn)
background_label.place(x=0,y=0,relwidth=1,relheight=1)
C.pack()
b2=Button(f1,text="QUIT              ",command=exitt,activebackground="red",font="1000",borderwidth="1",relief="solid",justify="center").place(x=100,y=700)
b1=Button(f1,text="START          ",command=run,activebackground="green",font="1000",borderwidth="1",justify="center",relief="solid").place(x=100,y=500)
l =Label(window,text="WELCOME TO PROCTOR MASTER",bg="lavender",fg="black",padx=150,pady=50,font=("comicsansms",19,"bold"),borderwidth=10,relief="groove").place(x=400,y=100)
sri = Label(window,text="Developed by Srijan Sachdeva,Ayush Rawat,Rinkesh Kumar,Piyush Yadav",bg="black",fg="white",padx=30,pady=30,font=("comicsansms",10,"bold"),borderwidth=10,relief="groove").place(x=965,y=700)
b3=Button(f1,text="INSTRUCTIONS",command=instruct,activebackground="orange",font="1000",borderwidth="1",justify="center",relief="solid").place(x=100,y=600)


window.mainloop()