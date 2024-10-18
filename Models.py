from huggingface_hub import HfApi
from datetime import datetime, timezone
import warnings
from concurrent.futures import ThreadPoolExecutor
from dateutil import parser
import os
from huggingface_hub.utils import disable_progress_bars
import random
import time
import json
import shutil
import re


class Models:
    """
    this class pulls the models from huggingface API and processes them
    """

    def __init__(self):
        """
        init method
        """
        self.api = HfApi()
        self.models = []
        self.model_by_time = []
        self.models_filtered = []
        warnings.filterwarnings("ignore", message="Invalid model-index")
        warnings.filterwarnings("ignore", message=".*HF_HUB_DISABLE_SYMLINKS_WARNING.*")
        disable_progress_bars()

    def get_models(self, limit=1000000, full=True):
        """
        pulls the models, with specified limit amount of models
        full=True makes filter by date possible
        """
        self.models = list(self.api.list_models(sort="createdAt", direction=-1, full=full))
        print("fetched %s models" % len(self.models))

    def write_models(self):
        models = []
        for model in self.models:
            models.append([model.modelId, model.createdAt])
        self.models = models
        with open('model_names.json', 'w') as file:
            for model in models:
                file.write(json.dumps(model) + '\n')

    def read_file(self, start_line, end_line, name):
        result = []
        with open(name, 'r') as file:
            for line_num, line in enumerate(file, start=start_line):
                if start_line <= line_num <= end_line:
                    data = json.loads(line.strip())
                    result.append(data)
                elif line_num > end_line:
                    return result
            return result

    def get_models_from_time(self, limit):
        start_date = datetime(2022, 9, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 8, 31, tzinfo=timezone.utc)
        page = 0
        page_size = 1000
        while len(self.model_by_time) < limit:
            start_line = page * page_size
            end_line = (page + 1) * page_size
            page += 1
            models = self.read_file(start_line, end_line, "model_names.json")
            if start_date <= parser.isoparse(models[0][1]) <= end_date:
                for model in models:
                    self.model_by_time.append(model)
            elif start_date >= parser.isoparse(models[0][1]):
                break

    def write_time_filtered_models(self):
        models = []
        for model in self.model_by_time:
            models.append(model[0])
        self.model_by_time = models
        with open('model_filtered_time.json', 'w') as file:
            for model in models:
                file.write(json.dumps(model) + '\n')

    def filter_date(self):
        """
        filter the models by date
        """
        start_date = datetime(2022, 9, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 8, 31, tzinfo=timezone.utc)
        filtered_models = [
            model for model in self.models
            if start_date <= parser.isoparse(str(model.createdAt)) <= end_date
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
            return model_info
        except Exception as error:
            return None

    def filter_empty(self):
        """
        filter the models with empty datacard out
        """
        with ThreadPoolExecutor(max_workers=100) as executor:
            percents = 10
            for i in range(0, percents):
                models = self.models[
                         round(len(self.models) * i / percents):round(len(self.models * (i + 1)) / percents)]
                model_ids = [model.modelId for model in models]
                results = list(executor.map(self.fetch_model_info, model_ids))
                models_filtered = [model for model, result in zip(models, results) if result is not None]
                for models in models_filtered:
                    self.models_filtered.append(models.id)
                print(len(models_filtered))
                print("%s percent done" % str((i + 1)))
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                if os.path.exists(cache_dir):
                    # shutil.rmtree(cache_dir)
                    print("Hugging Face cache has been cleared.")
                else:
                    print("Hugging Face cache directory does not exist.")
                time.sleep(3)

        # self.models = [model for model, result in zip(self.models, results) if result is not None]
        print("%s models have data cards" % len(self.models_filtered))

    def filter_empty_from_file(self):
        """
        filter the models with empty datacard out
        """
        page = 0
        page_size = 1000
        while page < 814:
            with ThreadPoolExecutor(max_workers=100) as executor:
                start_line = page * page_size
                end_line = (page + 1) * page_size
                page += 1
                models = self.read_file(start_line, end_line, "model_filtered_time.json")
                results = list(executor.map(self.fetch_model_info, models))

                models_filtered = [model for model, result in zip(models, results) if result is not None]
                for model in models_filtered:
                    self.models_filtered.append(model)
                if len(models_filtered) == 0:
                    print("RATE LIMIT")
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print("page:", page)
                    print("Hugging Face cache has been cleared.")
                else:
                    print("Hugging Face cache directory does not exist.")
                time.sleep(3)

    def filter_empty_card_from_file(self, threshold=100):
        """
        filter the models with empty datacard out
        """
        page = 0
        page_size = 1000
        count = 0
        while page < 612:
            with ThreadPoolExecutor(max_workers=100) as executor:
                start_line = page * page_size
                end_line = (page + 1) * page_size
                page += 1
                models = self.read_file(start_line, end_line, "model_with_card.json")
                results = list(executor.map(self.fetch_model_info, models))
                num = len(self.models_filtered)
                for result in results:
                    if result is not None:
                        with open(str(result), 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            content = re.sub(r"---(.*?)---", "", content, flags=re.DOTALL).strip()
                            if content:
                                match = re.search(r"models--(.*?)\\", result)
                                if match:
                                    self.models_filtered.append(match.group(1).replace("--", "/"))
                                if len(content) < threshold:
                                    count += 1

                if len(self.models_filtered) == num:
                    print("RATE LIMIT")
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print("page:", page)
                    print("Hugging Face cache has been cleared.")
                else:
                    print("Hugging Face cache directory does not exist.")
                time.sleep(10)
        return count

    def write_model_non_empty_cards(self):
        with open('model_with_non_empty_card.json', 'w') as file:
            for model in self.models_filtered:
                file.write(json.dumps(model) + '\n')

    def write_model_cards(self):
        with open('model_with_card.json', 'w') as file:
            for model in self.models_filtered:
                file.write(json.dumps(model) + '\n')

    def select_400(self):
        with open('400.txt', 'w') as file:
            random.shuffle(self.models_filtered)
            for i in range(0, 400):
                file.write(str(self.models_filtered[i]) + "\n")
        print("finished")


if __name__ == "__main__":
    md = Models()
    # count = md.filter_empty_card_from_file()
    # md.write_model_non_empty_cards()
    # print(count, "models have a non empty model card with less than 100 characters")
    print(sum(1 for line in open('model_with_card.json')))
    print(sum(1 for line in open('model_with_non_empty_card.json')))
