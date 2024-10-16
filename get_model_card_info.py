import huggingface_hub as hf

# Fetch model card details for a specific model
model_id = "google-bert/bert-base-uncased"  # Replace with your desired model

# Download the README.md (model card) file
readme_path = hf.hf_hub_download(model_id, 'README.md')

# Read and display the content of the README.md
with open(readme_path, 'r', encoding='utf-8') as file:
    model_card_content = file.read()

# Print out the model card (README content)
print("Model Card Content:")
print(model_card_content)