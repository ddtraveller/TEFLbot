import os
import sys
import gc
import psutil
import asyncio
import warnings
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This will force CPU usage

import speech_recognition as sr
import gtts
import pygame
import gpt4all
import langchain.text_splitter
import langchain_huggingface
import langchain_community.vectorstores
import chardet
import sklearn.feature_extraction.text
import sklearn.metrics.pairwise
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Create a ThreadPoolExecutor
thread_pool = ThreadPoolExecutor()

# Global variable for model path
model_path = "orca-mini-3b-gguf2-q4_0.gguf"

# Function to reinitialize key components
def reinitialize_components():
    recognizer = sr.Recognizer()
    model = initialize_gpt4all(model_path)
    embeddings = langchain_huggingface.HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Clear memory
    gc.collect()

    return recognizer, model, embeddings

def speak(text, lang='en', tld='co.uk', chunk_size=900):
    text = text.replace("'", "")
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    for chunk in chunks:
        tts = gtts.gTTS(text=chunk, lang=lang, tld=tld)
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

def listen(recognizer, timeout=10):
    try:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            print("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Listening...")
            try:
                audio = recognizer.listen(source, timeout=timeout)
            except sr.WaitTimeoutError:
                print("Listening timed out. No speech detected.")
                return None
        
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
        print(f"Microphone error: {str(e)}")
        return None

async def async_read_file_content(file_path):
    try:
        def read_file():
            with open(file_path, 'rb') as file:
                return file.read()
        
        raw_data = await asyncio.get_event_loop().run_in_executor(thread_pool, read_file)
        detected = await asyncio.get_event_loop().run_in_executor(thread_pool, chardet.detect, raw_data)
        encoding = detected['encoding'] or 'utf-8'
        
        content = raw_data.decode(encoding)
        del raw_data
        return content
    except Exception as e:
        print(f"Error reading {file_path}: {str(e)}")
        return ""

async def get_relevant_documents(query, doc_list, top_k=7):
    doc_contents = await asyncio.gather(*[async_read_file_content(doc) for doc in doc_list])
    non_empty_docs = [(doc, content) for doc, content in zip(doc_list, doc_contents) if content.strip()]
    
    if not non_empty_docs:
        print("No readable documents found.")
        return []
    
    filtered_docs, filtered_contents = zip(*non_empty_docs)
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(list(filtered_contents) + [query])
    cosine_similarities = sklearn.metrics.pairwise.cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    name_relevance = np.array([1 if any(term in doc.lower() for term in query.lower().split()) else 0 for doc in filtered_docs])
    combined_scores = cosine_similarities + name_relevance
    top_indices = np.argsort(combined_scores)[-top_k:][::-1]
    relevant_docs = [filtered_docs[i] for i in top_indices]
    
    print(f"Top {top_k} relevant documents found:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"{i}. {doc}")
    
    del doc_contents, non_empty_docs, filtered_docs, filtered_contents, vectorizer, tfidf_matrix, cosine_similarities, name_relevance, combined_scores, top_indices
    gc.collect()
    
    return relevant_docs

async def load_and_process_documents(file_paths, chunk_size=200, chunk_overlap=20):
    documents = []
    for file_path in file_paths:
        content = await async_read_file_content(file_path)
        if content:
            documents.append(content)
    
    if not documents:
        return []

    text_splitter = langchain.text_splitter.RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text("\n\n".join(documents))
    
    del documents, text_splitter
    gc.collect()
    
    return chunks

def initialize_gpt4all(model_path):
    print("Loading GPT4All model...")
    model = gpt4all.GPT4All(model_path)
    print("GPT4All model loaded.")
    return model

def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Current memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def generate_response(model, prompt, max_tokens=2000, continuation_attempts=3):
    full_response = ""
    for _ in range(continuation_attempts):
        with model.chat_session():
            response = model.generate(prompt=prompt, temp=0.7, max_tokens=max_tokens)
        full_response += response
        
        if response.strip().endswith(('.', '!', '?')):
            break
        else:
            prompt = f"{prompt}\n{response}\nPlease continue:"
    
    return full_response

def memory_cleanup():
    gc.collect()

async def process_question(question, model, embeddings, all_docs):
    context = ""
    if all_docs:
        try:
            relevant_docs = await get_relevant_documents(question, all_docs, top_k=7)
            texts = await load_and_process_documents(relevant_docs, chunk_size=200, chunk_overlap=20)
            
            if texts:
                vectorstore = langchain_community.vectorstores.FAISS.from_texts(texts, embeddings)
                context_docs = vectorstore.similarity_search(question, k=1)
                context = "\n".join([doc.page_content for doc in context_docs])
                del vectorstore, texts
        except Exception as e:
            print(f"Error processing documents: {e}")
    
    prompt = f"You are an AI assistant. Answer the following question:\n\nQuestion: {question}"
    if context:
        prompt = f"You are an AI assistant. Use this context to answer the question:\n\nContext:\n{context}\n\nQuestion: {question}"
    
    response = generate_response(model, prompt)
    memory_cleanup()
    return response

async def main():
    print("Starting script...")
    recognizer, model, embeddings = reinitialize_components()

    print("Select a role for the AI assistant:")
    print("1. Holistic Healing")
    print("2. Medical Herbalism")
    print("3. Permaculture")
    print("4. Software")
    print("5. General Topics (default)")

    role = input("Enter the number of your choice (or press Enter for General): ")

    base_path = os.path.abspath("TEFLTools")
    role_paths = {
        "1": ("Holistic Healing", os.path.join(base_path, "Readings", "Holistic_Healing_Arts_and_Practices")),
        "2": ("Medical Herbalism", os.path.join(base_path, "Readings", "Medical_Herbalism")),
        "3": ("Permaculture", os.path.join(base_path, "Readings", "Regenerative_Living")),
        "4": ("Software", os.path.join(base_path, "Readings", "Software_for_Timor_Leste")),
        "5": ("General Topics", base_path)
    }

    role_name, doc_path = role_paths.get(role, role_paths["5"])
    print(f"Selected role: {role_name}")

    all_docs = []
    if os.path.exists(doc_path):
        print(f"Scanning for documents in {doc_path}...")
        for root, _, files in os.walk(doc_path):
            for file in files:
                if file.endswith((".txt", ".md", ".pdf")) and "license" not in file.lower():
                    full_path = os.path.join(root, file)
                    if os.path.exists(full_path):
                        all_docs.append(full_path)
        print(f"Found {len(all_docs)} documents.")
    else:
        print("No document directory found. Continuing without documents.")

    is_first_iteration = True
    while True:
        try:
            if is_first_iteration:
                speak(f"What would you like to talk about regarding {role_name}?", lang='en', tld='co.uk')
                is_first_iteration = False
            else:
                recognizer, model, embeddings = reinitialize_components()
            
            log_memory_usage()
            question = listen(recognizer, timeout=10)
            
            if question:
                response = await process_question(question, model, embeddings, all_docs)
                print("Assistant:", response)
                speak(response, lang='en', tld='co.uk', chunk_size=600)
                del response
            else:
                speak("I didn't catch that. Could you repeat?", lang='en', tld='co.uk')

            gc.collect()
            time.sleep(0.1)

        except Exception as e:
            print(f"Error: {str(e)}")
            speak("I encountered an error. Let me restart.", lang='en', tld='co.uk')

        if psutil.virtual_memory().percent > 80:
            print("High memory usage. Restarting...")
            python = sys.executable
            os.execl(python, python, *sys.argv)

if __name__ == "__main__":
    asyncio.run(main())
