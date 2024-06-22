import speech_recognition as sr
from vosk import Model, KaldiRecognizer
import json
import pyttsx3
from gpt4all import GPT4All
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
import time
import chardet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    
    model = Model("path/to/vosk-model")  # Replace with the path to your Vosk model
    rec = KaldiRecognizer(model, 16000)
    
    rec.AcceptWaveform(audio.get_wav_data())
    result = rec.Result()
    text = json.loads(result)["text"]
    
    print(f"You: {text}\n")
    return text

print("Starting script...")

# Path to the GPT4All model file
model_path = "orca-mini-3b-gguf2-q4_0.gguf"

print("Loading GPT4All model...")
# Create a GPT4All model instance
model = GPT4All(model_path)
print("GPT4All model loaded.")

def read_file_content(file_path):
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
    doc_contents = [read_file_content(doc) for doc in doc_list]
    
    # Filter out empty documents
    non_empty_docs = [(doc, content) for doc, content in zip(doc_list, doc_contents) if content.strip()]
    
    if not non_empty_docs:
        print("No readable documents found.")
        return []
    
    filtered_docs, filtered_contents = zip(*non_empty_docs)
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(list(filtered_contents) + [query])
    
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    top_indices = np.argsort(cosine_similarities)[-top_k:][::-1]
    
    relevant_docs = [filtered_docs[i] for i in top_indices]
    
    print("Relevant documents found:")
    for doc in relevant_docs:
        print(f"- {doc}")
    
    return relevant_docs

def load_and_process_documents(file_paths, chunk_size=300, chunk_overlap=30):
    documents = []
    for file_path in file_paths:
        content = read_file_content(file_path)
        if content:
            documents.append(content)
    
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_text("\n\n".join(documents))

print("Scanning for documents in TEFLTools folder...")
tefl_tools_path = os.path.join(os.getcwd(), "TEFLTools")
all_docs = []
for root, _, files in os.walk(tefl_tools_path):
    for file in files:
        if file.endswith(".txt") and "license" not in file.lower():
            all_docs.append(os.path.join(root, file))
print(f"Found {len(all_docs)} documents in TEFLTools folder.")

print("Creating embeddings...")
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
print("Embeddings model loaded.")

is_first_iteration = True

print("Starting chat session...")
with model.chat_session():
    while True:
        if is_first_iteration:
            speak("What would you like to talk about?")
            is_first_iteration = False
        
        print("Listening for input...")
        question = listen()
        
        if question:
            print(f"Received question: {question}")
            
            try:
                print("Finding relevant documents...")
                relevant_docs = get_relevant_documents(question, all_docs)
                print(f"Found {len(relevant_docs)} relevant documents.")
                
                print("Loading and processing relevant documents...")
                texts = load_and_process_documents(relevant_docs, chunk_size=300, chunk_overlap=30)
                print(f"Processed {len(texts)} text chunks.")
                
                if not texts:
                    print("No text could be processed from the relevant documents. Using only the question for the response.")
                    context = ""
                else:
                    print("Creating vector store...")
                    embeddings = sentence_transformer.encode(texts)
                    vectorstore = FAISS.from_embeddings(list(zip(texts, embeddings)), sentence_transformer)
                    print("Vector store created.")
                    
                    print("Retrieving context...")
                    context_docs = vectorstore.similarity_search(question, k=2)
                    context = "\n".join([doc.page_content for doc in context_docs])
                
                print("Generating response...")
                enriched_prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                response = model.generate(prompt=enriched_prompt, temp=0.7, max_tokens=300)
                
                print("Friend:", response)
                speak(response)
            except Exception as e:
                error_message = f"An error occurred: {str(e)}\n"
                error_message += f"Error type: {type(e).__name__}\n"
                error_message += f"Error details: {str(e)}\n"
                error_message += f"Error occurred in: {e.__traceback__.tb_frame.f_code.co_filename}, line {e.__traceback__.tb_lineno}"
                print(error_message)
                speak("I'm sorry, but I encountered an error while processing your question. Could you please try asking something else?")
        else:
            speak("Sorry, I didn't catch that.")
            continue

        time.sleep(0.1)

    print(model.current_chat_session)