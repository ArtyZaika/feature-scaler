# feature scaler. homework task description

## Input data
The python-dev-test/data directory contains two files: test.tsv, train.tsv of the same format. Each line of the file is a characteristic for one vacancy.

The file has two columns:
* id_job - (integer) job ID
* features - (string) concatenation of features of a vacancy of a certain type (Combined through the symbol “,”). 
 ○ The first element of the list is the code of the feature set. (In this example, “2”)
* Other elements are numerical characteristics of this type. Within this task it is 256 integer numbers
(total each feature can be numbered and the name of the columns can look like: “feature_2_{i}”, where i is the index of the element in the array)(*).

## Requirements for output data
As a test, a script should be attached to the module that generates the test_proc.tsv file that contains the following set of columns (features) for each vacancy from test.tsv:
1. id_job - (integer) job ID (dimension: 1);
2. feature_2_stand_{i} - (double) standardization result (z-score normalization) of the input feature feature_2_{i} (See Input data(*)) (dimension : 256);
  - Reference to the definition of standardization:
    1. https://ru.wikipedia.org/wiki/Z-%D0%BE%D1%86%D0%B 5%D0%BD%D0%BA%D0%B0
    2. https://en.wikipedia.org/wiki/Feature_scaling
  - To perform this operation, it is required to evaluate two statistics for each feature_2_{i} column on the data from the train.tsv file:
    1. mean(feature_2_{i}) - the average value for all vacancies for feature_2_{i};
    2. std(feature_2_{i}) - standard deviation for all vacancies for feature_2_{i};
3. max_feature_2_index - (integer) index i of the maximum value of feature_2_{i} for this vacancy (dimension : 1);
4. max_feature_2_abs_mean_diff - (double) - absolute deviation of the feature with index max_feature_2_index from its mean value mean(feature_2_{max_feature_2_index}) (dimension : 1);

## Implementation requirements
1. The implementation must be done in Python 3.7 2. When designing a module, the following factors must be taken into account:
   1. Data sizes of the train.tsv, test.tsv file can reach several tens of millions of lines
   2. When using the module in the future, it is planned to add factors of new types (in our test, only signs of type “2”).
   3. When using the module in the future, it is planned to add new ways of feature normalization (in our test, only z-score normalization).
2. The package needs to be uploaded to the github repository.
