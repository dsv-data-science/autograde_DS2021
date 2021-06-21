# autograde_DS2021
Repository for the submission of Automatic Grading of Exam Responses: An Extensive Classification Benchmark, Discovery Science 2021.

The folder 'data' contains the Arabic dataset (AR-ASAG) that is reported in the paper. The DAMI dataset is protected under privacy, and we can therefore not publish it in the repository.

The folder 'notebooks' contains all experiments under notebooks>experiments. It also contains a brief data description under 'data_description', showcasing all the statistics of the AR-ASAG dataset found in the paper.

The folder 'utils' contains all functions that are used throughout the experiments. Here you can also find the RandomClassifier model under the sub-folder 'experiment_models'. RepeatedHoldOut is an abstract class of RepeatedBERT and RepeatedBaselines. These classes are wrappers that were created to perform repetitions of MCCV in an easy-to-read and easy-to-implement fashion.

All notebooks have already been executed for easy look-up on results.
