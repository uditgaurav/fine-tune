## Fine-Tune

Welcome to the Fine-Tune repository! This repository contains code for fine-tuning pre-trained language models using the Hugging Face Transformers library. Fine-tuning allows you to train a pre-trained language model on your specific task or dataset, which can lead to improved performance and better results.

### Usage
- The Fine-Tune repository provides easy-to-use scripts for fine-tuning pre-trained language models on your specific task or dataset. Here are the general steps to use Fine-Tune:

#### Clone the repository:
```bash
git clone https://github.com/uditgaurav/fine-tune.git
```
#### Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Ask a question
- To ask a question you first need to have  `OPENAI_API_KEY`

```bash
export OPENAI_API_KEY=ACBDEFGHIJKLMNOPQRSTUVWXYZ
```

- Now run `scripts/fine_tune.py` as shown below.

```bash
$> python3 scripts/fine_tune.py
Enter your question: why chaos engineering required?


Chaos engineering is required in order to ensure that systems are resilient to unexpected events. By deliberately introducing chaos into a system, engineers can identify weaknesses and potential points of failure. This helps to ensure that systems are able to withstand real-world conditions and can maintain stability even in the face of unexpected challenges.
```

#### What is the function or purpose of the script

This script is a Python code that uses the OpenAI API to generate text based on prompts and completions stored in a CSV file. The main steps in the script are as follows:

1. Import necessary libraries: The script imports the following libraries: openai, os, pandas, pickle, re, and fuzzywuzzy.

2. Set OpenAI API key: The script sets the API key for OpenAI by retrieving it from the environment variable OPENAI_API_KEY.

3. Load CSV data: The script loads data from a CSV file (prompts_and_completions.csv) into a pandas DataFrame (df), which contains prompts and completions.

4. Train and fine-tune models: The script iterates through the prompts and completions, and for each prompt, it checks if a trained model is already available. If not, it fine-tunes the model using the OpenAI API (openai.Completion.create()) with the specified parameters such as engine, prompt, max_tokens, temperature, n, and stop. The generated fine-tuned data is then processed and saved to a dictionary (trained_models) with the prompt as the key and the formatted response as the value.

5. Save trained models: The trained models are saved to a pickle file (trained_models.pkl) for future use.

6. Define text generation function: The script defines a function generate_text(prompt) that takes a prompt as input and generates text using the trained models. It uses fuzzy string matching (process.extractOne()) to find the closest match for the input prompt from the trained models based on similarity score. If the similarity score is above a certain threshold (80 in this case), it uses the corresponding trained model to generate text using the OpenAI API. Otherwise, it generates text by appending the prompt to a URL and using it as a new prompt for the OpenAI API.

7. Take user input and generate text: The script takes user input for a question prompt, calls the generate_text() function to generate text based on the input prompt, and then prints the generated text to the console.
