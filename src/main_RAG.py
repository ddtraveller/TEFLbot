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
    with sr.Microphone() as source:
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
        print(f"An unexpected error occurred: {str(e)}")
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

async def process_question(question, model, embeddings, all_docs):
    relevant_docs = await get_relevant_documents(question, all_docs, top_k=5)
    print(f"Found {len(relevant_docs)} relevant documents.")
    
    texts = await load_and_process_documents(relevant_docs, chunk_size=300, chunk_overlap=50)
    print(f"Processed {len(texts)} text chunks.")
    
    if not texts:
        print("No text could be processed from the relevant documents. Using only the question for the response.")
        context = ""
    else:
        vectorstore = langchain_community.vectorstores.FAISS.from_texts(texts, embeddings)
        context_docs = vectorstore.similarity_search(question, k=3)  # Increased from 1 to 3
        context = "\n".join([doc.page_content for doc in context_docs])
    
    enriched_prompt = f"""You are an AI assistant. Your primary task is to answer the question using ONLY the information provided in the following context. If the context doesn't contain enough information to fully answer the question, say so, but try to provide as much relevant information as possible from the context.

Context:
{context}

Question: {question}

Answer based ONLY on the above context:"""

    response = generate_response(model, enriched_prompt, temperature=0.3)  # Lowered temperature
    
    # Post-processing to verify response
    if context:
        context_words = set(context.lower().split())
        response_words = set(response.lower().split())
        overlap = len(context_words.intersection(response_words))
        if overlap < 5:  # Arbitrary threshold, adjust as needed
            response += "\n\nNote: This response may not be entirely based on the provided context. Please verify the information."
    
    del relevant_docs, texts, vectorstore, context, enriched_prompt
    memory_cleanup()
    
    return response

def generate_response(model, prompt, max_tokens=2000, continuation_attempts=3, temperature=0.3):
    full_response = ""
    for _ in range(continuation_attempts):
        with model.chat_session():
            response = model.generate(prompt=prompt, temp=temperature, max_tokens=max_tokens)
        full_response += response
        
        if response.strip().endswith(('.', '!', '?')):
            break
        else:
            prompt = f"{prompt}\n{response}\nPlease continue:"
    
    return full_response

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
    relevant_docs = await get_relevant_documents(question, all_docs, top_k=7)
    print(f"Found {len(relevant_docs)} relevant documents.")
    
    texts = await load_and_process_documents(relevant_docs, chunk_size=200, chunk_overlap=20)
    print(f"Processed {len(texts)} text chunks.")
    
    if not texts:
        print("No text could be processed from the relevant documents. Using only the question for the response.")
        context = ""
    else:
        vectorstore = langchain_community.vectorstores.FAISS.from_texts(texts, embeddings)
        context_docs = vectorstore.similarity_search(question, k=1)
        context = "\n".join([doc.page_content for doc in context_docs])
    
    enriched_prompt = f"You are an AI assistant. Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    response = generate_response(model, enriched_prompt)
    
    del relevant_docs, texts, vectorstore, context, enriched_prompt
    memory_cleanup()
    
    return response

async def main():
    print("Starting script...")
    
    # Initial setup
    recognizer, model, embeddings = reinitialize_components()

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

    is_first_iteration = True
    question_count = 0

    print("Starting chat session...")
    
    while True:
        try:
            if is_first_iteration:
                speak(f"What would you like to talk about regarding {role_name}?", lang='en', tld='co.uk')
                is_first_iteration = False
            else:
                print("Reinitializing components...")
                recognizer, model, embeddings = reinitialize_components()
                print("Reinitialization complete.")
            
            log_memory_usage()
            
            print("Listening for input...")
            question = listen(recognizer, timeout=10)
            
            if question:
                print(f"Received question: {question}")
                question_count += 1
                
                response = await process_question(question, model, embeddings, all_docs)
                
                print("Friend:", response)
                speak(response, lang='en', tld='co.uk', chunk_size=600)

                del response
                gc.collect()
            else:
                speak("I'm sorry, I didn't catch that. Could you please repeat your question?", lang='en', tld='co.uk')

            time.sleep(0.1)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n"
            error_message += f"Error type: {type(e).__name__}\n"
            error_message += f"Error details: {str(e)}\n"
            error_message += f"Error occurred in: {e.__traceback__.tb_frame.f_code.co_filename}, line {e.__traceback__.tb_lineno}"
            print(error_message)
            speak("I'm sorry, but I encountered an error. Let me restart and try again.", lang='en', tld='co.uk')

        log_memory_usage()

        # Check if memory usage is too high and restart if necessary
        if psutil.virtual_memory().percent > 80:
            print("Memory usage is too high. Restarting the script...")
            python = sys.executable
            os.execl(python, python, *sys.argv)

if __name__ == "__main__":
    asyncio.run(main())