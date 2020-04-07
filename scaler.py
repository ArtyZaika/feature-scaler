from typing import (Dict, Union, Callable, Iterable)
import numpy as np


class FeatureTransformer:
    """
    Feature Transformer.

    """
    FEATURES_LEN = 257

    def __init__(self, data_dir: str='data', test_file_name: str='test.tsv',
                 train_file_name: str='train.tsv', output_file_name: str='test_proc.tsv'):

        self.data_dir = data_dir
        self.test_file_name = test_file_name
        self.train_file_name = train_file_name
        self.output_file_name = output_file_name

        # stats fields
        self.features_mean = None
        self.features_std = None

    def _read_job_vacancy_line(self, is_train: bool=True):
        """
        Reads huge *.tsv files, yields one row at a time

        :param is_train: should train file be read or test
        :return: tuple of job_id and parsed features values list
        """
        file_name = self.train_file_name if is_train else self.test_file_name

        with open(self.data_dir + '/' + file_name) as f:
            for line_n, line in enumerate(f):
                if line_n == 0:
                    continue

                jobid_features = line.split('\t')
                job_id = jobid_features[0]
                features_values = list(map(int, jobid_features[1].split(',')))

                yield (job_id, features_values)

    def _calculate_train_feature_mean(self):
        """
        Reads 'train' file in order to calculate Mean(average) of each column in features list
        Note: summation of each column in features list made rowwise by Numpy, which make it very efficient.
        To avoid type overflow in Numpy np.int64 used for sums storing

        Time complexity: O(n), where n-size of train array
        :return:
        """
        fvalues_sum_dict = {}
        row_num = 0

        for row_num, (_, fvalues) in enumerate(self._read_job_vacancy_line()):
            if len(fvalues) != self.FEATURES_LEN:
                raise ValueError("features number should be equals to {}!".format(self.FEATURES_LEN))

            # feature's type should always be first element in feature's array
            features_type = fvalues[0]

            if features_type not in fvalues_sum_dict:
                # Note: to avoid type overflow in Numpy, np.int64 is used.
                # More here: https://numpy.org/devdocs/user/basics.types.html
                fvalues_sum_dict[features_type] = np.zeros((self.FEATURES_LEN-1,), dtype=np.int64)

            line_array = np.array(fvalues[1:self.FEATURES_LEN], dtype=np.int64)

            fvalues_sum_dict[features_type] += line_array

        # calculate mean over all feature_types
        for ftype_key in fvalues_sum_dict.keys():
            fvalues_sum_dict[ftype_key] = fvalues_sum_dict[ftype_key] / (row_num + 1)

        self.features_mean = fvalues_sum_dict

    def _calculate_train_feature_std(self, fvalues_avg: Dict[int, Iterable[float]]=None):
        """
        Reads 'train' file in order to calculate Std(Standard Deviation) of each column in features list
        Note: summation of each column in features list made rowwise by Numpy, which make it very efficient.
        To avoid type overflow in Numpy np.double used for sums storing

        Time complexity: O(n), where n-size of train array
        :return:
        """
        if not fvalues_avg:
            raise ValueError("fvalues_avg(mu) should not be None!")

        fvalues_diff_dict = {}
        row_num = 0

        for row_num, (_, fvalues) in enumerate(self._read_job_vacancy_line()):
            if len(fvalues) != self.FEATURES_LEN:
                raise ValueError("features number should be equals to {}!".format(self.FEATURES_LEN))

            # feature's type should always be first element in feature's array
            features_type = fvalues[0]

            if features_type not in fvalues_diff_dict:
                # Note: to avoid type overflow in Numpy, np.double is used.
                # More here: https://numpy.org/devdocs/user/basics.types.html
                fvalues_diff_dict[features_type] = np.zeros((self.FEATURES_LEN-1,), dtype=np.double)

            line_array = np.array(fvalues[1:self.FEATURES_LEN], dtype=np.int64)

            fvalues_diff_dict[features_type] += np.power(line_array - fvalues_avg[features_type], 2)

        # calculate std over all feature_types
        for ftype_key in fvalues_diff_dict.keys():
            fvalues_diff_dict[ftype_key] = np.sqrt(fvalues_diff_dict[ftype_key] / (row_num + 1))

        self.features_std = fvalues_diff_dict

    @staticmethod
    def _get_max_feature_index() -> Callable:
        cache = {'max_value': 0.0, 'max_value_index': 0}

        def decorate(index: int, value: int) -> Dict[str, Union[float, int]]:
            if value > cache['max_value']:
                cache['max_value'] = value
                cache['max_value_index'] = index

            return cache

        return decorate

    def fit(self):
        """
        Calculates Mean and Std on train dataset for using them in scaling

        Time complexity: O(n), where n-size of train array
        :return:
        """
        self._calculate_train_feature_mean()
        self._calculate_train_feature_std(self.features_mean)

    def transform(self):
        """
        1. Reads test dataset to transform its features
        2. Creates new features by transforming features from test dataset:
         - feature_{}_stand_{i} - (double), z-score normalization using statistics(Mean, Std) collected on 'fit(...)' step.
         - max_feature_{}_index - (integer) index i of max feature_2_{i} values for the job_id row (dimension : 1)
         - max_feature_2_abs_mean_diff - (double) - absolute deviation of a feature with index max_feature_{}_index
         from its average value mean(feature_{}_{max_feature_{}_index}). (dimension : 1)
        3. Saves transformed features to the out file. Operation made in place, after input row was read

        Time complexity: O(m*f), where m-size of test array f-features size
        :return:
        """
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

                features_type = fvalues[0]

                # init header of output features file, when feature's type parsed
                if not header:
                    header = ['id_job'] + \
                             ['feature_{0}_stand_{1}'.format(features_type, i) for i in range(1, self.FEATURES_LEN)] + \
                             ['max_feature_{0}_index'.format(features_type),
                              'max_feature_{0}_abs_mean_diff'.format(features_type)]

                    # write header to the file
                    header_line = '\t'.join(header)
                    f.write(header_line)
                    f.write("\n")

                for fvalue_index, fvalue in enumerate(fvalues):
                    if fvalue_index == 0:
                        continue

                    fvalue_int = int(fvalue)
                    index = fvalue_index-1

                    # seek for a max value index while iterating
                    max_value_index_cache = max_feature_index(index, fvalue_int)

                    # standard scale of a feature using Mean and Std calculated for train dataset
                    mean = self.features_mean[features_type][index]
                    std = self.features_std[features_type][index]

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
                    raise ValueError(" len of header should be equal to len of row values!"
                                     " Please check if each row has same features number")

                # write row values to the file
                row_line = '\t'.join(map(str, row_values))
                f.write(row_line)
                f.write("\n")



if __name__ == "__main__":
    f_scaler = FeatureTransformer()

    f_scaler.fit()
    f_scaler.transform()

