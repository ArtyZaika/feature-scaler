import unittest
from scaler import FeatureTransformer


class TestSum(unittest.TestCase):

    @unittest.skip
    def test_integration(self):
        train_file = 'test_test.tsv'
        test_file = 'test_test.tsv'
        outfile = 'test_test_proc.tsv'

        scaler = FeatureTransformer(data_dir='data_test', test_file_name=test_file, train_file_name=train_file,
                                    output_file_name=outfile)

        scaler.FEATURES_LEN = 5
        scaler.fit()
        scaler.transform()

        line_n = 0

        with open('data_test/' + outfile) as f:
            for line_n, line in enumerate(f):
                if line_n == 0:
                    header = line.split('\t')

                    # number of features should be equals
                    # 7: 1(job_id)+4(feature_2_stand_{i})+2(max_feature_2_index and max_feature_2_abs_mean_diff)
                    self.assertEqual(len(header), 7, "Should be 7")


                if line_n == 4:
                    features_values = line.split('\t')

                    self.assertEqual(len(features_values), 7, "Should be 7")
                    #id_job
                    self.assertEqual(int(features_values[0]), -4, "Should be -4")
                    # feature_2_stand_{i}
                    self.assertAlmostEqual(float(features_values[1]), 0.707, places=3)
                    self.assertAlmostEqual(float(features_values[2]), 0.707, places=3)
                    self.assertAlmostEqual(float(features_values[3]), 0.707, places=3)
                    self.assertAlmostEqual(float(features_values[4]), 0.707, places=3)
                    # max_feature_2_index
                    self.assertEqual(int(features_values[5]), 3, "Should be 3")
                    # max_feature_2_abs_mean_diff
                    self.assertEqual(float(features_values[6]), 1.0, "Should be 1.0")


        # output file should contain same amount of row as test file
        self.assertEqual(line_n+1, 6)

    @unittest.skip
    def test_test_file_corrupted(self):
        train_file = 'test_test.tsv'
        test_file = 'test_corrupted.tsv'
        outfile = 'test_test_proc.tsv'

        scaler = FeatureTransformer(data_dir='data_test', test_file_name=test_file, train_file_name=train_file,
                                    output_file_name=outfile)

        scaler.FEATURES_LEN = 5

        with self.assertRaises(ValueError) as context:
            scaler.fit()
            scaler.transform()

        self.assertTrue(' len of header should be equal to len of row values!'
                        ' Please check if each row has same features number' in str(context.exception))

    @unittest.skip
    def test_train_file_corrupted(self):
        train_file = 'test_train_corrupted.tsv'
        test_file = 'test_corrupted.tsv'
        outfile = 'test_test_proc.tsv'

        scaler = FeatureTransformer(data_dir='data_test', test_file_name=test_file, train_file_name=train_file,
                                    output_file_name=outfile)

        scaler.FEATURES_LEN = 5

        with self.assertRaises(ValueError) as context:
            scaler.fit()
            scaler.transform()

        self.assertTrue('features number should be equals to 5!' in str(context.exception))

    def test_transform_called_before_fit(self):
        train_file = 'test_test.tsv'
        test_file = 'test_test.tsv'
        outfile = 'test_test_proc.tsv'

        scaler = FeatureTransformer(data_dir='data_test', test_file_name=test_file, train_file_name=train_file,
                                    output_file_name=outfile)

        scaler.FEATURES_LEN = 5

        with self.assertRaises(ValueError) as context:
            scaler.transform()
            scaler.fit()

        self.assertTrue("You should train Scaler before using it. Run 'fit(...)' before 'transform(...)'!" in str(context.exception))



if __name__ == '__main__':
    unittest.main()