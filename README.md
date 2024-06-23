# TEFLbot
## Chatbot with GPT4All, Google Text-to-Speech, speech_recognition, and LangChain

This code creates a chatbot that leverages a local implementation of GPT4All to do the thinking, Google Text-to-Speech and speech_recognition to do the speaking and listening, and LangChain serving as a bridge to search local documents and sklearn to calculate the document relevance before sending it all to ChatGPT4All with pygame as an orchestrator.

The chatbot works as follows:

1. It looks through the local file system and loads up documents as context for the query to ChatGPT4 before the response is generated.
2. This means localized expert robots or even bots that can interrogate strange file structures can be easily created. You can point the bot at some log files, for example, and ask it if there is anything unusual about them.

### Limitations

- You can only load so many documents before the memory runs out.
- sklearn is used to try to check the relevance of documents before loading them, but you could just point the program to the documents you want and not worry about it.

The basic implementation in `main.py` will say, "What would you like to have a conversation about?" Then it will go into a listening mode and then try to respond. Finally, it will say, "Ask a question." and go into listening mode again.

To run:
```
python3 main.py
```

While the program speaks and listens, output on the command line looks like:
```
pygame 2.5.0 (SDL 2.28.0, Python 3.8.10)
Hello from the pygame community. https://www.pygame.org/contribute.html
Failed to load llamamodel-mainline-cuda-avxonly.dll: LoadLibraryExW failed with error 0x7e
Failed to load llamamodel-mainline-cuda.dll: LoadLibraryExW failed with error 0x7e
Listening...
Recognizing...
You: what is the name of this llm model

Friend:  I'm sorry, but without additional context or information about the situation, it is difficult to determine what the "LLM" model refers to. Can you please provide more details?
Listening...
Recognizing...
You: what is the capital city of Japan

Friend:  The capital city of Japan is Tokyo.
Listening...
Recognizing...
You: what are three great things about Tokyo

Friend:  As an AI assistant, I don't have personal opinions or beliefs. However, here are some interesting facts about Tokyo:

1. Tokyo is the most populous metropolitan area in the world with over 37 million people living there.
2. Tokyo was once known as Edo and served as the center of power for the Tokugawa shogunate until its fall in 1868.
3. Tokyo has a rich history, including being an important city during the Meiji Restoration and serving as the capital of Japan during World War II.
```

The `main_RAG.py` implementation looks through local files for files relevant to the user query and includes them in the context for the call to the LLM.

- `speech_recognition` (sr): This library can use online services for speech recognition. Specifically, when using `recognize_google()`, it sends audio data to Google's servers for processing.
- `gtts` (Google Text-to-Speech): This library makes API calls to Google's servers to generate speech from text.

# Guide to Tuning the AI Assistant Program - main_RAG.py

This guide will help you fine-tune the AI assistant program to achieve the desired balance between using the context from documents and the model's pre-existing knowledge.

## 1. Adjusting Relevant Document Selection

### Function: `get_relevant_documents(query, doc_list, top_k=5)`

The `top_k` parameter determines how many relevant documents are selected based on the query.

- **Increasing `top_k`**: 
  - Pro: Provides more context, potentially capturing more relevant information.
  - Con: May introduce noise and irrelevant information, increasing processing time.
- **Decreasing `top_k`**: 
  - Pro: Focuses on the most relevant documents, reducing noise.
  - Con: Might miss important information if it's spread across many documents.

Experiment with values between 3 and 10, depending on your document collection's size and diversity.

### Function: `vectorstore.similarity_search(question, k=3)`

This `k` value determines how many text chunks are retrieved from the vector store for context.

- **Increasing `k`**: 
  - Pro: Provides more detailed context.
  - Con: May introduce irrelevant information and increase token usage.
- **Decreasing `k`**: 
  - Pro: Focuses on the most relevant information.
  - Con: Might miss important details.

Try values between 1 and 5, balancing detail with relevance.

## 2. Adjusting Document Processing

### Function: `load_and_process_documents(file_paths, chunk_size=300, chunk_overlap=50)`

- **chunk_size**: Determines the length of text segments.
  - Increasing: Provides more context in each chunk but may reduce granularity.
  - Decreasing: Allows for more specific matching but may lose context.

- **chunk_overlap**: Controls how much text overlaps between chunks.
  - Increasing: Ensures context continuity but increases redundancy.
  - Decreasing: Reduces redundancy but may lose context between chunks.

Experiment with chunk sizes between 200 and 500, and overlaps between 20 and 100.

## 3. Modifying Model Temperature

### Function: `generate_response(model, prompt, max_tokens=2000, continuation_attempts=3, temperature=0.3)`

The `temperature` parameter controls the randomness of the model's output.

- **Increasing temperature** (e.g., 0.5-0.8): 
  - Pro: More creative and diverse responses.
  - Con: May stray from the given context.
- **Decreasing temperature** (e.g., 0.1-0.3): 
  - Pro: More focused and deterministic responses.
  - Con: May become repetitive or too constrained.

Adjust between 0.1 and 0.8, depending on how creative vs. deterministic you want the responses to be.

## 4. Post-processing Threshold

### In the `process_question` function:

```python
if overlap < 5:  # Arbitrary threshold, adjust as needed
    response += "\n\nNote: This response may not be entirely based on the provided context. Please verify the information."
```
This threshold checks how many words from the context appear in the response.

- **Increasing the threshold**:
    - Pro: Ensures responses are more closely tied to the context.
    - Con: May flag accurate responses that use different wording.
- **Decreasing the threshold**:
    - Pro: Allows for more flexibility in response wording.
    - Con: May not catch responses that stray from the context.
Experiment with values between 3 and 10, depending on your desired strictness.

## 5. Prompt Engineering
The current prompt is:
```enriched_prompt = f"""You are an AI assistant. Your primary task is to answer the question using ONLY the information provided in the following context. If the context doesn't contain enough information to fully answer the question, say so, but try to provide as much relevant information as possible from the context.

Context:
{context}

Question: {question}

Answer based ONLY on the above context:
```
# Prompt Components:
1. **Role Definition**: "You are an AI assistant."
2. **Task Instruction**: "Your primary task is to answer the question using ONLY the information provided in the following context."
3. **Handling Insufficient Information**: "If the context doesn't contain enough information to fully answer the question, say so, but try to provide as much relevant information as possible from the context."
4. **Context Presentation**: The relevant text is provided under "Context:"
5. **Question Presentation**: The user's question is clearly labeled.
6. **Final Instruction**: "Answer based ONLY on the above context:"

# Tuning the Prompt:
* **Specificity**: You can make the prompt more specific to your use case. For example, if it's for a medical assistant, you might add: "You are a medical AI assistant. Provide information based on scientific facts from the context."
* **Output Format**: You can specify a desired output format. For example: "Structure your answer in bullet points for clarity."
* **Encouraging Elaboration**: If you want more detailed answers, you might add: "Explain your reasoning and provide examples from the context if applicable."
* **Discouraging Speculation**: To further prevent the model from using its pre-existing knowledge, you could add: "Do not speculate or provide information beyond what is given in the context."
* **Handling Ambiguity**: You might add instructions for handling unclear questions: "If the question is ambiguous, explain the possible interpretations before answering."

Remember, prompt engineering is an iterative process. Test different prompts and observe how they affect the AI's responses in terms of relevance, accuracy, and adherence to the given context.

# Library Descriptions

## Standard Python Libraries

- **os**: Provides a way of using operating system dependent functionality.
- **sys**: Provides access to some variables used or maintained by the Python interpreter.
- **gc**: Garbage Collector interface.
- **time**: Time access and conversions.
- **warnings**: Warning control.
- **concurrent.futures**: Launching parallel tasks.
- **io.BytesIO**: In-memory bytes buffer.

## Third-party Libraries

- **psutil**: Cross-platform library for retrieving information on running processes and system utilization.
- **asyncio**: Asynchronous I/O, event loop, and coroutines.
- **speech_recognition**: Library for performing speech recognition with support for several engines and APIs.
- **gtts**: Google Text-to-Speech, a Python library and CLI tool to interface with Google Translate's text-to-speech API.
- **pygame**: Set of Python modules designed for writing video games, useful here for audio playback.
- **gpt4all**: Python bindings for the GPT4All chat interface.
- **langchain**: Library for building applications with large language models.
- **langchain_huggingface**: Hugging Face integration for LangChain.
- **langchain_community.vectorstores**: Vector stores for LangChain from the community.
- **chardet**: Universal encoding detector.
- **sklearn**: Machine learning library for Python.
- **numpy**: Fundamental package for scientific computing with Python.

## Environment Variables

- `CUDA_VISIBLE_DEVICES = '-1'`: Forces CPU usage instead of GPU.

# Installation Guide

This guide will walk you through the process of setting up the environment and installing all necessary libraries for the script.

## Prerequisites

- Python 3.8 or later
- pip (Python package installer)

## Step 1: Set up a virtual environment (optional but recommended)

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows, use: myenv\Scripts\activate
```
## Step 2: Install required libraries

pip install psutil
pip install SpeechRecognition
pip install gTTS
pip install pygame
pip install gpt4all
pip install langchain
pip install chardet
pip install scikit-learn
pip install numpy
pip install --upgrade langchain-community

## Step 3: Install PyAudio (required for speech_recognition)
On Windows:
- bashCopypip install pyaudio
On macOS:
- bashCopybrew install portaudio
- pip install pyaudio
On Linux:
- bashCopysudo apt-get install python3-pyaudio

## Step 4: Install Hugging Face Transformers
- bashCopypip install transformers

## Step 5: Install FAISS (Facebook AI Similarity Search)
- bashCopypip install faiss-cpu

## Step 6: Download the GPT4All model
- Download the orca-mini-3b-gguf2-q4_0.gguf model from the GPT4All website and place it in your project directory.



The `main_offline_only.py` implementation runs entirely offline. The impact on performance can vary, but here's a general breakdown of what you might expect:

**Speech Recognition:**
- Accuracy: Expect a 10-20% decrease in accuracy compared to cloud-based solutions like Google's speech recognition.
- Speed: Local processing might be 1.5-2x slower, depending on your hardware.

**Text-to-Speech:**
- Quality: Local TTS engines like pyttsx3 tend to sound more robotic. The difference in naturalness could be quite noticeable, perhaps a 30-40% decline in perceived quality.
- Speed: Generation speed might be similar or slightly slower than cloud-based solutions.

**Embeddings and Vector Search:**
- Quality: Local models like SentenceTransformer can be quite good. You might see only a 5-10% decrease in relevance compared to more advanced cloud models.
- Speed: Embedding generation and search could be 2-3x slower, especially on first run or with large document sets.

**Language Model (GPT4All):**
- Quality: Depending on the specific model used, you might see a 10-30% decrease in response quality compared to state-of-the-art cloud models.
- Speed: Generation speed could be 2-5x slower, heavily dependent on your hardware (especially GPU availability).

**Overall System Performance:**
- Latency: Expect the overall response time to increase by 50-100% due to all processing happening locally.
- Resource Usage: The application will use significantly more local CPU, RAM, and potentially GPU resources.

Change the `speak` function to change languages and voices.
```python
def speak(text, lang='en', tld='co.uk'):
```

Here are some examples of the available voices and accents:
```
English (US) - Female voice (default)
Language code: en
TLD: com

English (US) - Male voice
Language code: en
TLD: com.au

English (UK) - Female voice
Language code: en
TLD: co.uk

English (UK) - Male voice
Language code: en
TLD: co.uk (same as female voice)

French - Female voice
Language code: fr
TLD: fr

German - Female voice
Language code: de
TLD: de

Spanish - Female voice
Language code: es
TLD: es

Italian - Female voice
Language code: it
TLD: it

Japanese - Female voice
Language code: ja
TLD: jp

Korean - Female voice
Language code: ko
TLD: kr
```