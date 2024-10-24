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
import itertools
import shutil
import re


class Datasets:
    """
    This class pulls datasets from the Hugging Face API and processes them
    """

    def __init__(self):
        """
        Init method
        """
        self.api = HfApi()
        self.datasets = []
        self.datasets_by_time = []
        self.datasets_filtered = []
        warnings.filterwarnings("ignore", message="Invalid dataset-index")
        warnings.filterwarnings("ignore", message=".*HF_HUB_DISABLE_SYMLINKS_WARNING.*")
        disable_progress_bars()

    def get_datasets(self, full=True):
        """
        Pulls datasets, with a specified limit amount of datasets.
        full=True makes filter by date possible.
        """
        self.datasets = list(self.api.list_datasets(sort="createdAt", direction=-1, full=full))
        print(f"Fetched {len(self.datasets)} datasets")

    def write_datasets(self):
        datasets = []
        for dataset in self.datasets:
            datasets.append([dataset.id, dataset.createdAt])
        self.datasets = datasets
        with open('dataset_names.json', 'w') as file:
            for dataset in datasets:
                file.write(json.dumps(dataset) + '\n')

    def read_file(self, start_line, end_line, name):
        result = []
        with open(name, 'r') as file:
            lines = itertools.islice(file, start_line, end_line + 1)
            for line in lines:
                data = json.loads(line.strip())
                result.append(data)
            return result

    def filter_date(self):
        """
        filter the datasets by date
        """
        start_date = datetime(2022, 9, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 8, 31, tzinfo=timezone.utc)
        filtered_datasets = [
            dataset for dataset in self.datasets
            if start_date <= parser.isoparse(str(dataset.createdAt)) <= end_date
        ]
        self.datasets_by_time = filtered_datasets
        print("%s dataset is within the specified dates" % len(self.datasets))

    def get_datasets_from_time(self, limit):
        start_date = datetime(2022, 9, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 8, 31, tzinfo=timezone.utc)
        page = 0
        page_size = 1000
        while len(self.datasets_by_time) < limit:
            start_line = page * page_size
            end_line = (page + 1) * page_size
            page += 1
            datasets = self.read_file(start_line, end_line)
            print("page:", page)
            # print("current date:", parser.isoparse(datasets[0][1]))
            # print(start_date <= parser.isoparse(datasets[0][1]) <= end_date)
            # print(start_date >= parser.isoparse(datasets[0][1]))
            if start_date <= parser.isoparse(datasets[0][1]) <= end_date:
                for dataset in datasets:
                    self.datasets_by_time.append(dataset)
            elif start_date >= parser.isoparse(datasets[0][1]):
                break

    def write_time_filtered_datasets(self):
        datasets = []
        for dataset in self.datasets_by_time:
            datasets.append(dataset.id)
        self.datasets_by_time = datasets

        with open('dataset_filtered_time.txt', 'w') as file:
            for dataset in datasets:
                file.write(dataset + '\n')

    def fetch_dataset_info(self, id):
        try:
            result = self.api.hf_hub_download(id, 'README.md', repo_type='dataset')
            return result
        except Exception as error:
            return None

    def select_400(self):

        datasets = []

        # Open the text file in read mode to load datasets
        with open('selected_datasets.txt', 'r') as file:
            for line in file:
                line = line.strip().strip('"')  # Remove any leading/trailing whitespace
                if line:  # Check if the line is not empty
                    datasets.append(line)

        # Check if there are enough datasets to select from
        if len(datasets) < 400:
            print(f"Error: Not enough datasets available. Found {len(datasets)} datasets.")
            return

        selected = random.sample(datasets, 400)

        # Write the selected datasets to the output file
        with open('400_datasets.txt', 'w') as file:
            for i in selected:
                file.write(json.dumps(i) + "\n")  # Write each dataset as a JSON string

        print("Finished selecting 400 datasets")

    def filter_empty_card_from_file(self, threshold=100, page=0):
        """
        filter the models with empty datacard out
        """
        page_size = 1000
        count = 0
        while page < 190:
            with ThreadPoolExecutor(max_workers=100) as executor:
                start_line = page * page_size
                end_line = (page + 1) * page_size
                page += 1
                models = self.read_file(start_line, end_line, "dataset_filtered_time.json")
                results = list(executor.map(self.fetch_dataset_info, models))
                num = len(self.datasets_filtered)
                current=[]
                for result in results:
                    if result is not None:
                        with open(str(result), 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                            content = re.sub(r"---(.*?)---", "", content, flags=re.DOTALL).strip()
                            if content:
                                match = re.search(r"datasets--(.*?)\\", result)
                                if match:
                                    self.datasets_filtered.append(match.group(1).replace("--", "/"))
                                    current.append(match.group(1).replace("--", "/"))
                                if len(content) < threshold:
                                    count += 1

                if len(self.datasets_filtered) == num:
                    print("RATE LIMIT")
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
                    print("page:", page)
                    print(len(self.datasets_filtered))
                    print("Hugging Face cache has been cleared.")
                    self.append("dataset_with_non_empty_card.json",current)
                else:
                    print("Hugging Face cache directory does not exist.")
                time.sleep(5)
        return count

    def append(self, name, content):
        with open(name, 'a') as file:
            for dataset in content:
                file.write(json.dumps(dataset) + '\n')

    def write_dataset_non_empty_cards(self):
        with open('dataset_with_non_empty_card.json', 'w') as file:
            for dataset in self.datasets_filtered:
                file.write(json.dumps(dataset) + '\n')


if __name__ == "__main__":
    # 229325

    ds = Datasets()
    count = ds.filter_empty_card_from_file(page=163)

    print(count, "models have a non empty model card with less than 100 characters")
    # ds.get_datasets()
    # ds.write_datasets()
    # ds.get_datasets()
    # ds.filter_date()
    # ds.write_time_filtered_datasets()
    # print(sum(1 for line in open('dataset_filtered_time.txt')))
    # print(open("dataset_names.json"))
    # ds.get_datasets(limit=1000, full=True)
    # ds.write_datasets()

    # ds.get_datasets_from_time(9999999)
    # ds.write_time_filtered_datasets()
    #
    # ds.select(1000)
    #
    # ds.select_400()
