import speech_recognition as sr
from gtts import gTTS
from io import BytesIO
import pygame
from gpt4all import GPT4All
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import chardet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

"""
Library Imports:

speech_recognition (sr): Provides speech recognition capabilities, allowing the script to convert speech to text.
gtts (Google Text-to-Speech): Converts text to speech, enabling the script to speak responses.
BytesIO: Provides in-memory bytes buffer, used here for handling audio data.
pygame: A set of Python modules designed for writing video games, used here for audio playback.
gpt4all: Provides access to the GPT4All language model for generating responses.
re: Offers support for regular expressions in Python.
os: Provides a way to use operating system-dependent functionality like reading file paths.
langchain.text_splitter.RecursiveCharacterTextSplitter: Used for splitting large texts into smaller chunks.
langchain_huggingface.HuggingFaceEmbeddings: Provides text embedding capabilities using Hugging Face models.
langchain_community.vectorstores.FAISS: A vector store for efficient similarity search in high-dimensional spaces.
time: Provides various time-related functions.
chardet: Used for detecting character encoding of text.
sklearn.feature_extraction.text.TfidfVectorizer: Converts a collection of raw documents to a matrix of TF-IDF features.
sklearn.metrics.pairwise.cosine_similarity: Computes the cosine similarity between samples.
numpy (np): Provides support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions.
warnings: Provides functions to handle warning messages.

These libraries collectively enable speech recognition, text-to-speech conversion, natural language processing, 
document handling, and various utility functions needed for the chat application.
"""
"""
Library Dependencies:

Before running this script, ensure you have all the required libraries installed.
You can install them using pip with the following commands:

python3 -m pip install -r requirements.txt
or
python3 -m pip install SpeechRecognition
python3 -m pip install gTTS
python3 -m pip install pygame
python3 -m pip install gpt4all
python3 -m pip install langchain
python3 -m pip install langchain_huggingface
python3 -m pip install faiss-cpu
python3 -m pip install chardet
python3 -m pip install scikit-learn
python3 -m pip install numpy

Note: Some libraries might require additional system dependencies. For example:
- SpeechRecognition might require PyAudio, which can be installed with:
  python3 -m pip install PyAudio
  (On some systems, you might need to install portaudio before installing PyAudio)

- FAISS might have different versions for CPU and GPU. The command above installs the CPU version.
  For GPU support, you might need to install faiss-gpu instead.

- gpt4all might require additional steps for setup, depending on your system and the specific model you're using.

Always refer to the official documentation of each library for the most up-to-date installation instructions and 
any system-specific requirements.
"""
# Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def speak(text, lang='en', tld='co.uk'):
    """
    Convert text to speech and play it.

    Args:
    text (str): The text to be spoken.
    lang (str): The language of the text (default is English).
    tld (str): Top-level domain for the Google TTS service.

    Returns:
    None
    """
    text = text.replace("'", "")  # Remove apostrophes to avoid errors
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
    """
    Listen for audio input and convert it to text.

    Returns:
    str: The recognized text from the audio input, or None if recognition fails.
    """
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

print("Starting script...")

# Path to the GPT4All model file
model_path = "orca-mini-3b-gguf2-q4_0.gguf"

print("Loading GPT4All model...")
# Create a GPT4All model instance
model = GPT4All(model_path)
print("GPT4All model loaded.")

def read_file_content(file_path):
    """
    Read the content of a file with automatic encoding detection.

    Args:
    file_path (str): The path to the file to be read.

    Returns:
    str: The content of the file, or an empty string if reading fails.
    """
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read()
        detected = chardet.detect(raw_data)
        encoding = detected['encoding'] or 'utf-8'
        
        with open(file_path, 'r', encoding=encoding) as file:
            return file.read()
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return ""

def get_relevant_documents(query, doc_list, top_k=5):
    """
    Find the most relevant documents for a given query.

    Args:
    query (str): The query to match documents against.
    doc_list (list): List of document paths to search through.
    top_k (int): Number of top relevant documents to return.

    Returns:
    list: A list of the most relevant document paths.
    """
    doc_contents = [read_file_content(doc) for doc in doc_list]
    
    # Filter out empty documents
    non_empty_docs = [(doc, content) for doc, content in zip(doc_list, doc_contents) if content.strip()]
    
    if not non_empty_docs:
        print("No readable documents found.")
        return []
    
    filtered_docs, filtered_contents = zip(*non_empty_docs)
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(list(filtered_contents) + [query])
    
    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
    
    relevant_docs = [filtered_docs[i] for i in top_indices]
    
    print("Relevant documents found:")
    for doc in relevant_docs:
        print(f"- {doc}")
    
    return relevant_docs

def load_and_process_documents(file_paths, chunk_size=300, chunk_overlap=30):
    """
    Load and process documents into text chunks.

    Args:
    file_paths (list): List of file paths to process.
    chunk_size (int): The size of each text chunk.
    chunk_overlap (int): The overlap between chunks.

    Returns:
    list: A list of processed text chunks.
    """
    documents = []
    for file_path in file_paths:
        content = read_file_content(file_path)
        if content:
            documents.append(content)
    
    if not documents:
        return []

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text("\n\n".join(documents))

# Scan for documents in the TEFLTools folder
print("Scanning for documents in TEFLTools folder...")
tefl_tools_path = os.path.join(os.getcwd(), "TEFLTools")
all_docs = []
for root, _, files in os.walk(tefl_tools_path):
    for file in files:
        if file.endswith(".txt") and "license" not in file.lower():
            all_docs.append(os.path.join(root, file))
print(f"Found {len(all_docs)} documents in TEFLTools folder.")

# Create embeddings for document similarity search
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings created.")

is_first_iteration = True

print("Starting chat session...")
# Main chat loop
with model.chat_session():
    while True:
        if is_first_iteration:
            speak("What would you like to talk about?", lang='en', tld='co.uk')
            is_first_iteration = False
        
        print("Listening for input...")
        question = listen()
        
        if question:
            print(f"Received question: {question}")
            
            try:
                # Find relevant documents
                print("Finding relevant documents...")
                relevant_docs = get_relevant_documents(question, all_docs)
                print(f"Found {len(relevant_docs)} relevant documents.")
                
                # Process the relevant documents
                print("Loading and processing relevant documents...")
                texts = load_and_process_documents(relevant_docs, chunk_size=300, chunk_overlap=30)
                print(f"Processed {len(texts)} text chunks.")
                
                if not texts:
                    print("No text could be processed from the relevant documents. Using only the question for the response.")
                    context = ""
                else:
                    # Create vector store for similarity search
                    print("Creating vector store...")
                    vectorstore = FAISS.from_texts(texts, embeddings)
                    print("Vector store created.")
                    
                    # Retrieve relevant context
                    print("Retrieving context...")
                    context_docs = vectorstore.similarity_search(question, k=2)
                    context = "\n".join([doc.page_content for doc in context_docs])
                
                # Generate response using GPT4All model
                print("Generating response...")
                enriched_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                response = model.generate(prompt=enriched_prompt, temp=0.7, max_tokens=300)
                
                print("Friend:", response)
                speak(response, lang='en', tld='co.uk')
            except Exception as e:
                # Handle any errors that occur during processing
                error_message = f"An error occurred: {str(e)}\n"
                error_message += f"Error type: {type(e).__name__}\n"
                error_message += f"Error details: {str(e)}\n"
                error_message += f"Error occurred in: {e.__traceback__.tb_frame.f_code.co_filename}, line {e.__traceback__.tb_lineno}"
                print(error_message)
                speak("I'm sorry, but I encountered an error while processing your question. Could you please try asking something else?", lang='en', tld='co.uk')
        else:
            speak("Sorry, I didn't catch that.", lang='en', tld='co.uk')
            continue

        time.sleep(0.1)

    print(model.current_chat_session)