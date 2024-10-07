from huggingface_hub import HfApi

api = HfApi()

model_card = api.model_info(repo_id="bert-base-uncased")

dataset_card = api.dataset_info(repo_id="bert-base-uncased")