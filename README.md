# CD38_Predictor

XGB regressor model to predict expression of CD38 (a protein of interest in the diagnosis of Multiple Myeloma and other Hematological cancers).

Permutation Importance is utilized for feature selection from a list of 100 known transcription factors of interest.

Original data maybe aquired by registering at https://research.themmrf.org/ Note that we utilized IA13.

Transcriptomic data from the COMMPASS study (https://themmrf.org/finding-a-cure/our-work/the-mmrf-commpass-study/ Interim Analysis 13). The hundred transcription factors were selected based on experiments conducted on cell lines. We developed both Random Forest models, as well as using Xgboost. Both sets of models had feature engineering techniques applied to them. One is a wrapper method known as permutation importance and the other was correlate based.

Initially all transcription factors determined to be co expressed were utilized together to build the model.  We used the standard hyperparameters for Random Forests used in the sklearn package, but chose to use randomized search with cross validation to find the best parameters for the XGBoost models. The results of the hyperparameter search were what were used for all subsequent models built.

For both, random forest, and XGBoost models, three were built. One with all transcription factors (which was also used to determine optimal hyperparameters for XGB as noted above). One using the top twenty highest correlated transcription factors (with CD38 based using spearman rank), and one using the top twenty transcription factors as determined by permutation importance (which is where each transcription factors value is randomized in turn, and if there is a loss in model performance it is seen as the feature carrying valuable information).
