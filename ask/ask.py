import os
import openai
import pandas as pd

# Set your OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Read the CSV file
df = pd.read_csv('prompts_and_completions.csv')

# Train the fine-tuned data using prompts and completions together
fine_tuned_data = []
for index, row in df.iterrows():
    prompt = row['prompt']
    completion = row['completion']
    fine_tuned_data.append(f"{prompt} {completion}")

# Write the fine-tuned data to a file
with open('fine_tuned_data.txt', 'w') as f:
    f.write('\n'.join(fine_tuned_data))

# Take user input for a question
user_question = input("Enter your question: ")

# Find the corresponding fine-tuned data for the user question
found = False
for index, prompt in enumerate(df['prompt']):
    if prompt.lower() in user_question.lower():
        fine_tuned_completion = df['completion'][index]
        found = True
        break

if not found:
    print("No matching prompt found in the fine-tuned data.")
else:
    print("Answer: ", fine_tuned_completion)
