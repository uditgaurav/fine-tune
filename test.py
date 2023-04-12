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
    if prompt in trained_models:
        trained_model = trained_models[prompt]
        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=prompt,
            max_tokens=1024,
            temperature=0.5,
            n=1,
            stop=None
        )
        generated_text = response['choices'][0]['text']
        return generated_text
    else:
        return f"No trained model found for prompt: {prompt}"

# Example usage of the generate_text() function
prompt = input("Enter your question: ")
generated_text = generate_text(prompt)
print("Generated text:", generated_text)
