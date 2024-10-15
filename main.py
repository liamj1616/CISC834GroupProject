from Models import *
from huggingface_hub import HfApi
import re
import requests

api = HfApi()

dataset_card = api.dataset_info(repo_id="imdb")


# model_info = api.model_info(repo_id="google-bert/bert-base-uncased")

# model_card_content = model_info.card_data

# print(dir(model_card_content))

# print(len(model_card_content))

# if model_card_content:

#     section_titles = re.findall(r'^(#{1,6})\s+(.+)', model_card_content, re.MULTILINE)


#     for level, title in section_titles:
#         print(f"{level} {title}")

# else:
#     print("No model card content available")


def fetch_section_titles(url):
    # Fetch the content of the model card
    response = requests.get(url)
    section_titles = []

    # Check if the request was successful
    if response.status_code == 200:
        model_card_content = response.text

        # Extract section titles from the markdown content
        section_titles = re.findall(r'^(#{1,6})\s+(.+)', model_card_content, re.MULTILINE)

    return section_titles


print(fetch_section_titles("https://huggingface.co/bert-base-uncased/raw/main/README.md"))
print(fetch_section_titles("https://huggingface.co/datasets/imdb/raw/main/README.md"))



