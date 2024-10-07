import csv
import numpy as np
from itertools import zip_longest


class DataProcessor:
    """
    this class uses the data that is gathered from
    the DataCollector class and then processes them,
    this class also have a method to plot the data
    """

    def __init__(self, csv_c_models="data/classification_models.csv",
                 csv_c_spaces="data/classification_spaces.csv",
                 csv_g_models="data/generation_models.csv",
                 csv_g_spaces="data/generation_spaces.csv"):
        """
        initialization,
        stores the file locations and the goal of processing the data
        :param csv_c_models: the file location for models of classification
        :param csv_c_spaces: the file location for spaces of classification
        :param csv_g_models: the file location for models of generation
        :param csv_g_spaces: the file location for spaces of generation
        """
        self.csv_classification_models = csv_c_models
        self.csv_classification_spaces = csv_c_spaces
        self.csv_generation_models = csv_g_models
        self.csv_generation_spaces = csv_g_spaces
        self.average_size_classification = -1
        self.average_size_generation = -1
        self.average_file_amount_classification = -1
        self.average_file_amount_generation = -1
        self.median_size_classification = -1
        self.median_size_generation = -1
        self.median_amount_classification = -1
        self.median_amount_generation = -1
        # self.classification_likes = []
        # self.generation_likes = []
        # self.classification_downloads = []
        # self.generation_downloads = []
        self.classification_size = []
        self.generation_size = []
        self.classification_amount = []
        self.generation_amount = []

    def get_data(self):
        """
        this method geths the data from the files
        :return:
        """

        c_size = []
        g_size = []
        c_amount = []
        g_amount = []

        # get data from space files
        with open(self.csv_classification_spaces) as classification_space_file:
            classification_space = csv.reader(classification_space_file)
            next(classification_space)
            for row in classification_space:
                c_size.append(int(row[3]))
                c_amount.append(int(row[4]))

        with open(self.csv_generation_spaces) as generation_space_file:
            generation_space = csv.reader(generation_space_file)
            next(generation_space)
            for row in generation_space:
                g_size.append(int(row[3]))
                g_amount.append(int(row[4]))

        # store the values
        self.classification_size = c_size.copy()
        self.generation_size = g_size.copy()
        self.classification_amount = c_amount.copy()
        self.generation_amount = g_amount.copy()

    def process(self, csv_save="data/processed_data.csv"):
        """
        this method performs calculation on the data,
        including average and median, no return but writes in the csv file
        :return:
        """
        # calculate the average
        self.average_size_classification = \
            sum(self.classification_size) / len(self.classification_size)
        self.average_size_generation = \
            sum(self.generation_size) / len(self.generation_size)
        self.average_file_amount_classification = \
            sum(self.classification_amount) / len(self.classification_amount)
        self.average_file_amount_generation = \
            sum(self.generation_amount) / len(self.generation_amount)

        # calculate the median
        self.median_size_classification = np.median(self.classification_size)
        self.median_size_generation = np.median(self.generation_size)
        self.median_amount_classification = \
            np.median(self.classification_amount)
        self.median_amount_generation = np.median(self.generation_amount)

        # storing the results in a csv file
        with open(csv_save, "w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                ["classification size", "generation size",
                 "average classification size", "average generation size",
                 "median classification size", "median generation size",
                 "classification amount", "generation amount",
                 "average classification amount", "average generation amount",
                 "median classification amount", "median generation amount"])
            for row in zip_longest(self.classification_size,
                                   self.generation_size,
                                   [self.average_size_classification],
                                   [self.average_size_generation],
                                   [self.median_size_classification],
                                   [self.median_size_generation],
                                   self.classification_amount,
                                   self.generation_amount,
                                   [self.average_file_amount_classification],
                                   [self.average_file_amount_generation],
                                   [self.median_amount_classification],
                                   [self.median_amount_generation],
                                   fillvalue=""):
                csv_writer.writerow(row)


if __name__ == "__main__":
    # Processes the data and displays the plot
    data = DataProcessor()
    data.get_data()
    data.process()
