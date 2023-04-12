from fuzzywuzzy import process
import openai
import os
import pandas as pd
import pickle

# Set your OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Load CSV data into a pandas DataFrame
df = pd.read_csv('prompts_and_completions.csv')

# Access the prompt and completion columns
prompts = df['prompt']
completions = df['completion']

# Load trained models from the pickle file
with open('trained_models.pkl', 'rb') as f:
    trained_models = pickle.load(f)

# Define a function to generate text using trained models
def generate_text(prompt):
    closest_match = process.extractOne(prompt, trained_models.keys())
    closest_prompt = closest_match[0]
    similarity = closest_match[1]
    if similarity > 80:  # Set a similarity threshold for matching prompts
        trained_model = trained_models[closest_prompt]
        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=closest_prompt,
            max_tokens=1024,
            temperature=0.5,
            n=1,
            stop=None
        )
        generated_text = response['choices'][0]['text']
    else:
        url = "https://developer.harness.io/docs/chaos-engineering/"
        prompt = f"Read the following document from {url} and answer the following question:\n\n{prompt}\n\nAnswer:"

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=1024,
            temperature=0.5,
            n=1,
            stop=None
        )
        generated_text = response['choices'][0]['text']
    return generated_text


prompt = input("Enter your question: ")
generated_text = generate_text(prompt)
print("Generated text:", generated_text)