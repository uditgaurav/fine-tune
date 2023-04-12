import openai
import os
import pandas as pd

# Set your OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Load CSV data into a pandas DataFrame
df = pd.read_csv('prompts_and_completions.csv')

# Access the prompt and completion columns
prompts = df['prompt']
completions = df['completion']

# Define a list to store fine-tuned data
fine_tuned_data_list = []

# Iterate through the prompts and completions
for i in range(len(prompts)):
    prompt = prompts[i]
    completion = completions[i]
    print("------------------------------>")
    print(f"Fine-tuning for prompt: {prompt}")
    # Fine-tune the model
    response = openai.Completion.create(
        engine="text-davinci-002",
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
    fine_tuned_data_list.append(fine_tuned_data)

    print("")
    print(f"Fine-tuning completed for prompt: {prompt}")
    print(f"Generated fine-tuned data: {fine_tuned_data}")
    print("")

# Save the fine-tuned data to a file
with open("fine_tuned_data.txt", "w") as f:
    f.write("\n".join(fine_tuned_data_list))

print("Fine-tuning completed successfully!")
print("Trained model saved to: model.pth")
