# TEFLbot
A python program that can perform conversational English that can be deployed on minimal hardware.
<br/> 
The current hello world implementation will say, "What would you like to have a conversation about?"
Then it will go into a listening mode and then try to respond.
Finally, it will say, "Ask a question." and go into listening mode again.
<br/> 
To run:<br/> 
python3 main.py()<br/> 
While the program speaks and listens, output on the command line looks like;<br/> 
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
<br/> 
Change the speak function to change languages and voices.<br/> 
```
def speak(text, lang='en', tld='co.uk'):
```
Here are some examples of the available voices and accents:<br/> 
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
