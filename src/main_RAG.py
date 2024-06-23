import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This will force CPU usage

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
import gc

warnings.filterwarnings("ignore", category=FutureWarning)

def speak(text, lang='en', tld='co.uk', chunk_size=900):
    """
    Convert text to speech and play it, chunking longer texts.
    """
    text = text.replace("'", "")  # Remove apostrophes to avoid errors
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    for chunk in chunks:
        tts = gTTS(text=chunk, lang=lang, tld=tld)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        pygame.mixer.init()
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

def initialize_recognizer():
    return sr.Recognizer()

def listen(recognizer):
    """
    Listen for audio input and convert it to text.
    """
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        print("Recognizing...")
        query = recognizer.recognize_google(audio, language='en-in')
        print(f"You: {query}\n")
        return query
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand that.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

def read_file_content(file_path):
    """
    Read the content of a file with automatic encoding detection.
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

def get_relevant_documents(query, doc_list, top_k=10):
    """
    Find the most relevant documents for a given query.
    """
    doc_contents = [read_file_content(doc) for doc in doc_list]
    non_empty_docs = [(doc, content) for doc, content in zip(doc_list, doc_contents) if content.strip()]
    
    if not non_empty_docs:
        print("No readable documents found.")
        return []
    
    filtered_docs, filtered_contents = zip(*non_empty_docs)
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(list(filtered_contents) + [query])
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    name_relevance = np.array([1 if any(term in doc.lower() for term in query.lower().split()) else 0 for doc in filtered_docs])
    combined_scores = cosine_similarities + name_relevance
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    relevant_docs = [filtered_docs[i] for i in top_indices]
    
    print(f"Top {top_k} relevant documents found:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"{i}. {doc}")
    
    return relevant_docs

def load_and_process_documents(file_paths, chunk_size=300, chunk_overlap=30):
    """
    Load and process documents into text chunks.
    """
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

def initialize_gpt4all(model_path):
    print("Loading GPT4All model...")
    model = GPT4All(model_path)
    print("GPT4All model loaded.")
    return model

def main():
    print("Starting script...")

    model_path = "orca-mini-3b-gguf2-q4_0.gguf"
    model = initialize_gpt4all(model_path)
    recognizer = initialize_recognizer()

    # Role selection
    print("Select a role for the AI assistant:")
    print("1. Holistic Healing")
    print("2. Medical Herbalism")
    print("3. Permaculture")
    print("4. Software")
    print("5. General Topics (default)")

    role = input("Enter the number of your choice (or press Enter for General): ")

    if role == "1":
        role_name = "Holistic Healing"
        doc_path = "./TEFLTools/Readings/Holistic_Healing_Arts_and_Practices"
    elif role == "2":
        role_name = "Medical Herbalism"
        doc_path = "./TEFLTools/Readings/Medical_Herbalism"
    elif role == "3":
        role_name = "Permaculture"
        doc_path = "./TEFLTools/Readings/Regenerative_Living"
    elif role == "4":
        role_name = "Software"
        doc_path = "./TEFLTools/Readings/Software_for_Timor_Leste"
    else:
        role_name = "General Topics"
        doc_path = "./TEFLTools"

    print(f"Selected role: {role_name}")

    print(f"Scanning for documents in {doc_path} folder...")
    all_docs = []
    for root, _, files in os.walk(doc_path):
        for file in files:
            if file.endswith(".txt") and "license" not in file.lower():
                all_docs.append(os.path.join(root, file))
    print(f"Found {len(all_docs)} documents in {role_name} folder.")

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embeddings created.")

    is_first_iteration = True
    question_count = 0

    print("Starting chat session...")
    while True:
        if is_first_iteration:
            speak(f"What would you like to talk about regarding {role_name}?", lang='en', tld='co.uk')
            is_first_iteration = False
        else:
            if 'texts' in locals():
                del texts
            if 'vectorstore' in locals():
                del vectorstore
            gc.collect()
            print("Unloaded previous documents from memory.")
        
        print("Listening for input...")
        question = listen(recognizer)
        
        if question:
            print(f"Received question: {question}")
            question_count += 1
            
            try:
                print("Finding relevant documents...")
                relevant_docs = get_relevant_documents(question, all_docs, top_k=10)
                print(f"Found {len(relevant_docs)} relevant documents.")
                
                print("Loading and processing relevant documents...")
                texts = load_and_process_documents(relevant_docs, chunk_size=300, chunk_overlap=30)
                print(f"Processed {len(texts)} text chunks.")
                
                if not texts:
                    print("No text could be processed from the relevant documents. Using only the question for the response.")
                    context = ""
                else:
                    print("Creating vector store...")
                    vectorstore = FAISS.from_texts(texts, embeddings)
                    print("Vector store created.")
                    
                    print("Retrieving context...")
                    context_docs = vectorstore.similarity_search(question, k=2)
                    context = "\n".join([doc.page_content for doc in context_docs])
                
                print("Generating response...")
                enriched_prompt = f"You are an AI assistant specialized in {role_name}. Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
                with model.chat_session():
                    response = model.generate(prompt=enriched_prompt, temp=0.7, max_tokens=1000)
                
                print("Friend:", response)
                speak(response, lang='en', tld='co.uk', chunk_size=600)
            except Exception as e:
                error_message = f"An error occurred: {str(e)}\n"
                error_message += f"Error type: {type(e).__name__}\n"
                error_message += f"Error details: {str(e)}\n"
                error_message += f"Error occurred in: {e.__traceback__.tb_frame.f_code.co_filename}, line {e.__traceback__.tb_lineno}"
                print(error_message)
                speak("I'm sorry, but I encountered an error while processing your question. Could you please try asking something else?", lang='en', tld='co.uk')
        else:
            speak("I'm sorry, I didn't catch that. Could you please repeat your question?", lang='en', tld='co.uk')

        if question_count % 2 == 0:
            print("Reinitializing GPT4All model and speech recognizer...")
            del model
            del recognizer
            gc.collect()
            model = initialize_gpt4all(model_path)
            recognizer = initialize_recognizer()

        time.sleep(0.1)

if __name__ == "__main__":
    main()