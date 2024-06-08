import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pygame
from gpt4all import GPT4All
import re

def speak(text, lang='en', tld='com'):
    # Replace apostrophes with empty strings
    text = text.replace("'", "")
    tts = gTTS(text=text, lang=lang, tld=tld)
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        print(f"You: {query}\n")
        return query
    except Exception as e:
        print("Error:", str(e))
        return None

model_path = "orca-mini-3b-gguf2-q4_0.gguf"
model = GPT4All(model_path)

with model.chat_session():
    while True:
        speak("What would you like to talk about?")
        question = listen()
        if question:
            response = model.generate(prompt=question, temp=0)
            print("Friend:", response)
            speak(response, lang='en', tld='co.uk')  # Change the voice to a male voice
        else:
            speak("Sorry, I didn't catch that. Please try again.")

    print(model.current_chat_session)
