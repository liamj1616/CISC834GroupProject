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
import requests


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
        disable_progress_bars()

    def get_datasets(self, limit=1000000, full=True):
        """
        Pulls datasets, with a specified limit amount of datasets.
        full=True makes filter by date possible.
        """
        self.datasets = list(self.api.list_datasets(sort="createdAt", direction=-1, full=full))
        print(f"Fetched {len(self.datasets)} datasets")

    def write_datasets(self):
        datasets = []
        for dataset in self.datasets:
            
            created_at = dataset.created_at
            if isinstance(created_at, datetime):
                created_at = created_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            datasets.append([dataset.id, created_at])
            
        print(datasets)
        
        self.datasets = datasets
        with open('dataset_names.json', 'w') as file:
            for dataset in datasets:
                file.write(json.dumps(dataset) + '\n')

    def read_file(self, start_line, end_line):
        result = []
        with open("dataset_names.json", 'r') as file:
            for line_num, line in enumerate(file, start=1):
                if start_line <= line_num <= end_line:
                    data = json.loads(line.strip())
                    result.append(list(data))
                elif line_num > end_line:
                    return result

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
            if start_date <= parser.isoparse(datasets[0][1]) <= end_date:
                for dataset in datasets:
                    self.datasets_by_time.append(dataset)
            elif start_date >= parser.isoparse(datasets[0][1]):
                break

    def write_time_filtered_datasets(self):
        datasets = []
        for dataset in self.datasets_by_time:
            datasets.append(dataset[0])
        self.datasets_by_time = datasets
        
        with open('dataset_filtered_time.txt', 'w') as file:
            for dataset in datasets:
                file.write(dataset + '\n')

    def filter_empty(self):
        """
        Filter datasets with empty data cards.
        """
        with ThreadPoolExecutor(max_workers=100) as executor:
            percents = 10
            for i in range(0, percents):
                datasets = self.datasets[
                           round(len(self.datasets) * i / percents):round(len(self.datasets * (i + 1)) / percents)]
                dataset_ids = [dataset.id for dataset in datasets]
                results = list(executor.map(self.fetch_dataset_info, dataset_ids))
                datasets_filtered = [dataset for dataset, result in zip(datasets, results) if result is not None]
                for dataset in datasets_filtered:
                    self.datasets_filtered.append(dataset.id)
                print(len(datasets_filtered))
                print(f"{(i + 1) * 10}% done")
                cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
                if os.path.exists(cache_dir):
                    print("Hugging Face cache has been cleared.")
                else:
                    print("Hugging Face cache directory does not exist.")
                time.sleep(3)

        print(f"{len(self.datasets_filtered)} datasets have data cards")
        
    def select(self, max):
        datasets = []
        # Open the text file in read mode to load datasets
        with open('dataset_filtered_time.txt', 'r') as file:
            for line in file:
                line = line.strip()  # Remove any leading/trailing whitespace
                if line:  # Check if the line is not empty
                    try:
                        dataset = json.loads(line)  # Parse each line as JSON
                        datasets.append(dataset)  # Add to the datasets list
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line: {line}. Error: {e}")
        # Check if there are enough datasets to select from
        if len(datasets) < max:
            print(f"Error: Not enough datasets available. Found {len(datasets)} datasets.")
            return
        # Shuffle the datasets
        random.shuffle(datasets)
        selected = datasets[:max]
        # Write the selected datasets to the output file
        # with open('selected_datasets.txt', 'w') as file:
        #     for i in selected:
        #         file.write(json.dumps(i) + "\n")  # Write each dataset as a JSON string

        print(f"Finished selecting {max} datasets")
        
        filtered_models = []
        
        for i in selected:
            if self.check_dataset_card(i):
                filtered_models.append(i)
                
        with open('selected_datasets.txt', 'w') as file:
            for dataset in filtered_models:
                file.write(dataset + '\n')

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

        # Shuffle the datasets
        random.shuffle(datasets)

        # Select the first 400 datasets
        selected = datasets[:400]

        # Write the selected datasets to the output file
        with open('400_datasets.txt', 'w') as file:
            for i in selected:
                file.write(json.dumps(i) + "\n")  # Write each dataset as a JSON string

        print("Finished selecting 400 datasets")
        
    def check_dataset_card(self, dataset_name):
        # Base URL for the Hugging Face API to get model details
        url = f"https://huggingface.co/api/datasets/{dataset_name}"
    
        try:
            # Send a request to the Hugging Face API
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            dataset_info = response.json()

            # Check if the dataset has a card by looking for a "cardData" or similar key
            return dataset_info.get("cardData") is not None

        except requests.exceptions.RequestException as e:
            print(f"Error fetching dataset data: {e}")
            return False


if __name__ == "__main__":
    ds = Datasets()
    
    # ds.get_datasets(limit=1000, full=True)
    # ds.write_datasets()
    
    # ds.get_datasets_from_time(9999999)
    # ds.write_time_filtered_datasets()
    
    # ds.select(1000)
    
    ds.select_400()