# autograde_DS2021
Repository for the submission of Automatic Grading of Exam Responses: An Extensive Classification Benchmark, Discovery Science 2021.

The folder 'data' contains the Arabic dataset (AR-ASAG) that is reported in the paper. The DAMI dataset is protected under privacy, and we can therefore not publish it in the repository.

The folder 'notebooks' contains all experiments under notebooks>experiments. It also contains a brief data description under 'data_description', showcasing all the statistics of the AR-ASAG dataset found in the paper. The following notebooks can be found in sub-folder 'experiments':
- flatten_experiment: contains the code and the results of the flatten experiment for all models.
- loqo_experiment: contains the code and the results of the loqo experiment for all models.
- qbased_experiment: contains the code and the results of the qbased experiment for all models.

The folder 'utils' contains the following files:
- configs.py: contains the global configurations. That is, the random seed used throughout the code.
- utils.py: contains all global functions that has been used throughout the experiments.
- sub-folder 'experiment_models': here you can find the RandomClassifier model that was used for the paper under RandomClassifier.py. RepeatedHoldOut.py is an abstract class of RepeatedBERT.py and RepeatedBaselines.py. These classes are wrappers that were created to perform repetitions of MCCV in an easy-to-read and easy-to-implement fashion. RepeatedBaselines is a wrapper of MCCV using sklearn implemented algorithms such as LogisticRegression and RandomForestClassifier. RepeatedBERT is a wrapper of the ktrain library, making MCCV implementation easier. For examples of its usage, see the experiments under the folder 'notebooks'.

All notebooks have already been executed for easy look-up on results.
