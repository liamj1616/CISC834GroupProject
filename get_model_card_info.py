from huggingface_hub import HfApi, hf_hub_download, model_info
from datetime import datetime, timezone
import huggingface_hub as hf

class ModelFilter:
    def __init__(self, category, start_date, end_date):
        self.api = HfApi()
        self.category = category
        # Make start_date and end_date timezone-aware
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        
        self.sections_to_check = [
            "Model Details",
            "Training Data",
            "Intended Use",
            "Evaluation Data",
            "Versioning",
            "Acknowledgements",
            "Citation Details",
            "Domain Specific",
            "Metrics",
            "General Data",
            "Ethical Considerations",
            "Caveats And Recommendations"
        ]
        self.section_tally = {section: 0 for section in self.sections_to_check}

    def filter_models_by_task(self):
        # List models filtered by task category
        models = list(self.api.list_models(filter=self.category, sort="downloads", direction=-1, limit=1000, full=True))
        
        print("fetched %s models" % len(models))
        
        return models

    def filter_models_by_date(self, models):
        # Filter models based on the creation date
        filtered_models = []
        for model in models:
            model_info_data = model_info(model.modelId)

            # Access the 'created_at' field directly from the ModelInfo object
            if hasattr(model_info_data, 'created_at') and model_info_data.created_at:
                creation_date = model_info_data.created_at
                
                # Ensure creation_date is within the provided date range
                if self.start_date <= creation_date <= self.end_date:
                    
                    filtered_models.append(model.modelId)
                    
        print(f"ALL MODELS RETRIEVED {len(filtered_models)}")
        print(filtered_models)

        return filtered_models
    
    def check_sections(self, readme_content):
        # Check for the presence of specified sections in the README content
        for section in self.sections_to_check:
            if section in readme_content:
                self.section_tally[section] += 1

    def run(self):
        # First, filter by task category
        task_filtered_models = self.filter_models_by_task()
        
        # Then, filter by date range
        date_filtered_models = self.filter_models_by_date(task_filtered_models)
        
        for model_id in date_filtered_models:
            
            # print(f"Fetching README for {model_id}...")
            
            readme_path = hf.hf_hub_download(model_id, 'README.md')

            # Read and display the content of the README.md
            with open(readme_path, 'r', encoding='utf-8') as file:
                model_card_content = file.read()
                
            if model_card_content:
                # print(f"Model Card Content for {model_id}:\n")
                # print(model_card_content)
                # print("\n" + "="*50 + "\n")
                
            # Check sections in the README content
                self.check_sections(model_card_content)

        # Print out the section tally
        print("Section Tally:")
        for section, count in self.section_tally.items():
            print(f"{section}: {count}")

# Example usage
model_filter = ModelFilter(category="text-generation", start_date="2024-07-01", end_date="2024-08-1")
model_filter.run()
