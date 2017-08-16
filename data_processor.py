import numpy as np
from PIL import Image


class DataProcessor:
    DATA_FOLDER = 'data'
    DATA_LABELS = 'dataLabels.csv'
    TOTAL_DATA = 50000

    def __init__(self):
        self.input = None
        self.output = None
        self.num_labels = 0
        self.label_index_to_name_dict = {}
        self.label_name_to_index_dict = {}

    def load_input(self):
        # load images
        data = []
        for i in xrange(1, self.TOTAL_DATA + 1):
            img = Image.open('%s/%s.png' % (self.DATA_FOLDER, i))
            rgb_arr = np.array(img)
            rgb_arr = rgb_arr.flatten()

            data.append(rgb_arr)

        data = np.matrix(data)
        self.input = data

    def load_output(self):
        # load labels
        with open(self.DATA_LABELS) as f:
            data = [line.strip().split(',') for line in f if line]
            # remove headers
            data = data[1:]
            # turn image numbers into ints
            data = map(lambda x: (int(x[0]), x[1]), data)
            data.sort()
            # extract label names
            data = [i[1] for i in data]
            # turn labels into indices
            unique_names = set(data)
            for i, name in enumerate(unique_names):
                self.label_index_to_name_dict[i] = name
                self.label_name_to_index_dict[name] = i

            data = [self.label_name_to_index_dict[name] \
                    for name in data]

            self.output = np.matrix(data)
            self.output = self.output.transpose()
            self.num_labels = len(unique_names)

    @staticmethod
    def normalize_features(X, has_intercept_term):
        if has_intercept_term:
            # remove first column of 1s
            X = np.delete(X, 0, 1)
        # use Z-score normalization
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X -= mean
        X /= std
        if has_intercept_term:
            # add back first column of 1s
            X = np.insert(X, 0, 1, 1)
        return X

    def split_to_training_test(self, ratio):
        m = len(self.output)
        training = int(ratio * m)
        self.training_input = self.input[:training]
        self.training_output = self.output[:training]
        self.test_input = self.input[training:]
        self.test_output = self.output[training:]

