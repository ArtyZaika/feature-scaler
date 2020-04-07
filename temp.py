from collections import defaultdict
from math import sqrt
from typing import Dict
import numpy as np

FEATURES_LEN = 256

class FeatureScaler:

    def __init__(self, data_dir: str='data', test_file_name: str='test.tsv',
                 train_file_name: str='train.tsv', output_file_name: str='test_proc.tsv'):

        self.data_dir = data_dir
        self.test_file_name = test_file_name
        self.train_file_name = train_file_name
        self.output_file_name = output_file_name

        # stats fields
        self.row_numbers = 0
        self.features_mean = None
        self.features_std = None

    def _read_job_vacancy_line(self, is_train=True):
        self.row_numbers = 0
        file_name = self.train_file_name if is_train else self.test_file_name

        with open(self.data_dir + '/' + file_name) as f:
            for line_n, line in enumerate(f):
                if line_n == 0:
                    continue

                jobid_features = line.split('\t')
                job_id = jobid_features[0]
                features_values = jobid_features[1].split(',')

                self.row_numbers = line_n

                yield (job_id, features_values)

    def _get_train_feature_sum(self, func, func_acc_dict: Dict):
        for _, fvalues in self._read_job_vacancy_line():
            features_type = 0
            for fvalue_index, fvalue in enumerate(fvalues):

                # feature's type should be parsed but not calculated
                if fvalue_index == 0:
                    features_type = fvalue
                    continue

                fvalue_int = int(fvalue)

                func(func_acc_dict, features_type, fvalue_index, fvalue_int)

            if not fvalue_index != 256:
                raise ValueError("features number should be equals to 256!")


    @staticmethod
    def _get_train_average_feature_sum(acc_dict: Dict, n: int, sqrt_scale: bool=False):
        for f_type in acc_dict.keys():
            for i, fkey in enumerate(acc_dict[f_type].keys()):
                if sqrt_scale:
                    acc_dict[f_type][fkey] = sqrt(acc_dict[f_type][fkey] / n)
                else:
                    acc_dict[f_type][fkey] = acc_dict[f_type][fkey] / n

        return acc_dict


    def _calculate_train_mean(self):
        def calculate_sum(acc_dict: Dict, f_type:int, index: int, value: float):
            if f_type not in acc_dict:
                acc_dict[f_type] = defaultdict(int)

            acc_dict[f_type][index] += value

        fvalues_mean = {}

        self._get_train_feature_sum(calculate_sum, fvalues_mean)
        row_nums = self.row_numbers
        fvalues_mean = self._get_train_average_feature_sum(fvalues_mean, row_nums)

        self.features_mean = fvalues_mean


    def _calculate_train_std(self, fvalues_avg=None):
        if not fvalues_avg:
            raise ValueError("fvalues_avg(mu) should not be None!")

        def calculate_sum(acc_dict: Dict, f_type:int, index: int, value: float):
            if f_type not in acc_dict:
                acc_dict[f_type] = defaultdict(int)

            acc_dict[f_type][index] += (value - fvalues_avg[f_type][index]) ** 2

        fvalues_std = {}

        self._get_train_feature_sum(calculate_sum, fvalues_std)
        row_nums = self.row_numbers
        fvalues_std = self._get_train_average_feature_sum(fvalues_std, row_nums, True)

        self.features_std = fvalues_std

    @staticmethod
    def _get_max_feature_index():
        cache = {'max_value': 0.0, 'max_value_index': 0}

        def decorate(index: int, value: int) -> Dict[float, int]:
            if value > cache['max_value']:
                cache['max_value'] = value
                cache['max_value_index'] = index

            return cache

        return decorate

    def fit(self):
        self._calculate_train_mean()
        self._calculate_train_std(self.features_mean)

    def transform(self):
        if not self.features_mean or not self.features_std:
            raise ValueError("You should train Scaler before using it. Run 'fit(...)' before 'transform(...)'!")

        # header of output features file
        header = None

        # due to size of input file output will be written as new line read in test file
        with open(self.data_dir + '/' + self.output_file_name, 'w+') as f:
            for id_job, fvalues in self._read_job_vacancy_line(is_train=False):
                # values of each riw in output file
                row_values = [id_job]

                # memoization for max value index calculation
                max_feature_index = self._get_max_feature_index()

                features_type = 0

                for fvalue_index, fvalue in enumerate(fvalues):
                    if fvalue_index == 0:
                        features_type = fvalue

                        # init header of output features file, when feature's type parsed
                        if not header:
                            header = ['id_job'] +\
                                     ['feature_{0}_stand_{1}'.format(features_type, i) for i in range(1, 257)] +\
                                     ['max_feature_{0}_index'.format(features_type),
                                      'max_feature_{0}_abs_mean_diff'.format(features_type)]

                            # write header to the file
                            header_line = '\t'.join(header)
                            f.write(header_line)
                            f.write("\n")

                        continue

                    fvalue_int = int(fvalue)

                    # seek for a max value index while iterating
                    max_value_index_cache = max_feature_index(fvalue_index, fvalue_int)

                    # standard scale of a feature using Mean and Std calculated for train dataset
                    mean = self.features_mean[features_type][fvalue_index]
                    std = self.features_std[features_type][fvalue_index]

                    feature_z_scaled = (fvalue_int - mean) / std
                    row_values.append(feature_z_scaled)


                # index of max value of a feature
                max_value = max_value_index_cache['max_value']
                max_value_index = max_value_index_cache['max_value_index']
                row_values.append(max_value_index)

                # max feature's deviation between max value and avg value
                abs_mean_diff = abs(max_value - self.features_mean[features_type][max_value_index])
                row_values.append(abs_mean_diff)

                if len(header) != len(row_values):
                    raise ValueError("len of header should be equal to len of row values!")

                # write row values to the file
                row_line = '\t'.join(map(str, row_values))
                f.write(row_line)
                f.write("\n")



if __name__ == "__main__":
    f_scaler = FeatureScaler()
    f_scaler.fit()

    f_scaler.transform()

