from fuzzywuzzy import fuzz
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
    if similarity > 70:  # Set a similarity threshold for matching prompts
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
        return generated_text
    else:
        return f"No trained model found for prompt: {prompt}. Closest match: {closest_prompt}"

# Example usage of the generate_text() function
prompt = input("Enter your question: ")
generated_text = generate_text(prompt)
print("Generated text:", generated_text)
