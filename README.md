# TEFLbot
A python program that can perform conversational English that can be deployed on minimal hardware.

The current hello world implementation will say, "What would you like to have a conversation about?"
Then it will go into a listening mode and then try to respond.
Finally, it will say, "Ask a question." and go into listening mode again.

To run:
python3 main.py()
Output on the command line looks like;
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
