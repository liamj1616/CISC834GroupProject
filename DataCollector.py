from huggingface_hub import HfApi
import csv


class DataCollector:
    """
    this class gathers data from huggingface
    then processes them and save the processed data into csv files
    this class utilizes HfApi to access the data from huggingface
    """

    def __init__(self):
        """
        initialization, create api object
        """
        # access the API
        self.api = HfApi()

    def access_models(self, task, sort="downloads", direction=-1, limit=20):
        """
        this method accesses the models section of huggingface,
        through the use of the API
        :param task: the task(filter) it performs
        :param sort: the sorting of the models,
        can be "downloads" or "likes" for the purpose of this task
        :param direction: the direction of the sort
        :param limit: the max amount of returned models
        :return: returns nothing but writes to the csv file
        """

        # initializing csv file location
        csv_name = "data/" + task + "_models.csv"
        # initializing the csv file with titles
        with open(csv_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "Author", "Name", "Downloads", "Likes"])

        # get models from huggingface
        models = self.api.list_models(
            filter="text-" + task,
            sort=sort, direction=direction, limit=limit)
        for model_data in models:
            # for each model, get the id, author, name, downloads, and likes
            model_id = model_data.id
            id_split = model_id.split("/")
            author = id_split[0]
            model_name = id_split[1]
            downloads = str(model_data.downloads)
            likes = str(model_data.likes)

            # store above data into the csv file
            with open(csv_name, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    [model_id, author, model_name, downloads, likes])

    def access_spaces(self, task, limit=20):
        """
        this method accesses the spaces section of huggingface,
        through the use of the API
        :param task: the task(filter) it performs
        :param limit: the max amount of returned spaces
        :return: returns nothing but writes to the csv file
        """
        # initializing csv file location

        csv_name = "data/" + task + "_spaces.csv"
        csv_models = "data/" + task + "_models.csv"
        # initializing the csv file
        with open(csv_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ["ID", "Author", "Name", "app.py size", "num files"])
        try:
            with open(csv_models) as model_file:
                # get the data from the model file
                csv_reader = csv.reader(model_file)

                for row in csv_reader:
                    model_name = row[2]  # 2 is the name of the model

                    if model_name != "Name":  # if it's not the first row

                        # search for all spaces with that name
                        # the list_spaces() function
                        # doesn't work with models=_ like how it describes
                        spaces = self.api.list_spaces(
                            search=model_name, limit=limit)
                        for space in spaces:  # for each space in spaces
                            # space info is needed here because
                            # list_spaces() doesn't give enough data
                            space_info = self.api.space_info(
                                repo_id=space.id, files_metadata=True)

                            # for each space, get id, author,
                            # name of the space, app.py's size,
                            # and the amount of files
                            space_id = space_info.id
                            id_split = space_id.split("/")
                            author = id_split[0]
                            space_name = id_split[1]
                            siblings = space_info.siblings
                            for sibling in siblings:
                                if sibling.rfilename == 'app.py':
                                    app_size = sibling.size
                            num_files = len(siblings)
                            # store above data into csv file
                            with open(csv_name, "a", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerow(
                                    [space_id, author,
                                     space_name, app_size, num_files])
        except FileNotFoundError:
            print("models file is not found")


if __name__ == "__main__":
    # gathers all the data and store them in file
    data = DataCollector()
    data.access_models("generation")
    print("checkpoint1")
    data.access_spaces("generation")
    print("checkpoint2")
    data.access_models("classification")
    print("checkpoint3")
    data.access_spaces("classification")
    print("checkpoint4")
