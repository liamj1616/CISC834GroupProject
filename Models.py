from huggingface_hub import HfApi
from datetime import datetime, timezone
import warnings
from concurrent.futures import ThreadPoolExecutor


class Models:
    """
    this class pulls the models from huggingface API and processes them
    """

    def __init__(self):
        """
        init method
        """
        self.api = HfApi()
        self.models = None
        warnings.filterwarnings("ignore", message="Invalid model-index")
        warnings.filterwarnings("ignore", message=".*HF_HUB_DISABLE_SYMLINKS_WARNING.*")

    def get_models(self, limit=1000000, full=True):
        """
        pulls the models, with specified limit amount of models
        full=True makes filter by date possible
        """
        self.models = list(self.api.list_models(sort="downloads", direction=-1, limit=limit, full=full))
        print("fetched %s models" % len(self.models))

    def filter_date(self):
        """
        filter the models by date
        """
        start_date = datetime(2022, 9, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 8, 31, tzinfo=timezone.utc)
        filtered_models = [
            model for model in self.models
            if start_date <= datetime.fromisoformat(str(model.lastModified)) <= end_date
        ]
        self.models = filtered_models
        print("%s models is within the specified dates" % len(self.models))

    def fetch_model_info(self, model_id):
        """
        you need model info to filter if the model data card exists
        this method gets the model info given model id
        """
        try:
            model_info = self.api.hf_hub_download(model_id, 'README.md')
            print(model_info)
            return model_info
        except Exception as error:
            return None

    def filter_empty(self):
        """
        filter the models with empty datacard out
        """
        with ThreadPoolExecutor(max_workers=100) as executor:
            model_ids = [model.modelId for model in self.models]
            results = list(executor.map(self.fetch_model_info, model_ids))

        self.models = [model for model, result in zip(self.models, results) if result is not None]
        print("%s models have data cards" % len(self.models))


if __name__ == "__main__":
    md = Models()
    md.get_models(limit=100, full=True)
    # md.filter_date()
    md.filter_empty()
