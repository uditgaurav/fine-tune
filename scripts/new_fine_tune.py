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

# Define a dictionary to store trained models
trained_models = {}

# Load trained models from file, if available
if os.path.exists("trained_models.pkl"):
    with open("trained_models.pkl", "rb") as f:
        trained_models = pickle.load(f)

# Iterate through the prompts and completions
for i in range(len(prompts)):
    prompt = prompts[i]
    completion = completions[i]
    print("------------------------------>")
    print(f"Fine-tuning for prompt: {prompt}")
    
    # Check if trained model is already available
    if prompt in trained_models:
        print("Trained model already available. Loading...")
        trained_model = trained_models[prompt]
    else:
        # Fine-tune the model
        response = openai.Completion.create(
            engine="text-davinci-001",  # Update engine name
            prompt=prompt,
            max_tokens=1024,
            temperature=0.5,
            n=1,
            stop=None
        )

        # Extract the generated fine-tuned data
        fine_tuned_data = response['choices'][0]['text']

        # Process and save the fine-tuned data
        # You can save the fine-tuned data to a file, upload it to a storage service, or use it as needed in your workflow
        trained_model = fine_tuned_data
        trained_models[prompt] = trained_model

        print("")
        print(f"Fine-tuning completed for prompt: {prompt}")
        print(f"Generated fine-tuned data: {fine_tuned_data}")
        print("")

    # Use the trained model as needed in your workflow

# Save the trained models to a file
with open("trained_models.pkl", "wb") as f:
    pickle.dump(trained_models, f)

print("Fine-tuning completed successfully!")
print("Trained models saved to: trained_models.pkl")
