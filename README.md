# autograde_DS2021
Repository for the submission of Automatic Grading of Exam Responses: An Extensive Classification Benchmark, Discovery Science 2021

The folder 'data' contains the arabic dataset (AR-ASAG) that is reported in the paper. The DAMI dataset is protected under privacy, and we can therefore not publish it in the repository.

The folder 'notebooks' contains all experiments under notebooks>experiments. It also contains a brief data description under 'data_description', showcasing all the statistics found in the paper. All files in this folder are juypter notebooks.

The folder 'utils' contains all functions that are used throughout the experiments. Here you can also find the RandomClassifier model under the sub-folder 'experiment_models'. RepeatedHoldOut is an abstract class of RepeatedBERT and RepeatedBaselines. These classes are wrappers that was created in order to perform repetitions of MCCV in a easy to read and easy to implement fashion for the jupyter notebooks in 'notebooks'.

All notebooks have been ran for easy look-up on results.
