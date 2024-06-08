import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pygame
from gpt4all import GPT4All
import re

def speak(text, lang='en', tld='com'):
    """
    Convert text to speech and play the audio.

    Args:
        text (str): The text to be converted to speech.
        lang (str): The language code for the speech (default: 'en').
        tld (str): The top-level domain for the Google Text-to-Speech API (default: 'com').
    """
    # Replace apostrophes with empty strings
    text = text.replace("'", "")
    
    # Create a gTTS object with the given text, language, and top-level domain
    tts = gTTS(text=text, lang=lang, tld=tld)
    
    # Create a BytesIO object to store the audio data
    fp = BytesIO()
    
    # Write the audio data to the BytesIO object
    tts.write_to_fp(fp)
    
    # Move the file pointer to the beginning of the BytesIO object
    fp.seek(0)
    
    # Initialize the pygame mixer
    pygame.mixer.init()
    
    # Load the audio data from the BytesIO object
    pygame.mixer.music.load(fp)
    
    # Play the audio
    pygame.mixer.music.play()
    
    # Wait until the audio finishes playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

def listen():
    """
    Listen for user input using the microphone and perform speech recognition.

    Returns:
        str: The recognized text from the user's speech.
        None: If an error occurs during speech recognition.
    """
    # Create a Recognizer object
    r = sr.Recognizer()
    
    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    
    try:
        print("Recognizing...")
        # Perform speech recognition using Google Speech Recognition
        query = r.recognize_google(audio, language='en-in')
        print(f"You: {query}\n")
        return query
    except Exception as e:
        print("Error:", str(e))
        return None

# Path to the GPT4All model file
model_path = "orca-mini-3b-gguf2-q4_0.gguf"

# Create a GPT4All model instance
model = GPT4All(model_path)

# Start a chat session with the GPT4All model
with model.chat_session():
    while True:
        # Prompt the user to speak
        speak("What would you like to talk about?")
        
        # Listen for user input
        question = listen()
        
        if question:
            # Generate a response using the GPT4All model
            response = model.generate(prompt=question, temp=0)
            
            # Print the generated response
            print("Friend:", response)
            
            # Speak the generated response with a male voice
            speak(response, lang='en', tld='co.uk')
        else:
            # Handle the case when no user input is detected
            speak("Sorry, I didn't catch that. Please try again.")

    # Print the entire chat session history
    print(model.current_chat_session)