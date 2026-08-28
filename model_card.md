# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a supervised binary classification model to predict whether an individual's annual income is greater than $50,000 or less than or equal to $50,000. The model was trained using the U.S. Census Income dataset. Categorical features are transformed using one-hot encoding before training
## Intended Use
The model is intended as an educational machine learning project demonstrating data preprocessing, classification, model evaluation, performance monitoring across data slices, and deployment through a REST API. It is not intended to make real-world employment, lending, insuarance, or other high-impact decisions.
## Training Data
The model was trained using the Census Income dataset containing 32,561 observations and 15 columns. The target variable is salary, with two classes: <=50K. The dataset was split into training and testing sets, with 80% of the data used for training and 20% used for testing. Categorical features were processed using one-hot encoding.
## Evaluation Data
Twenty percent of the Census Income dataset was reserved as the test set using a reproducible random split with random_state=42. The salary variable was used for stratification so that the class distribution was preserved between the training and testing datasets. The test data was not used to train the model.
## Metrics
The model was evaluated using precision, recall, and F1 score. On the test dataset, the model achieved a precision of 0.7353, recall of 0.6378, and F1 score of 0.6831. Performance was also evaluated across individual across individual categorical data slices to identify differences in model performance among subgroups.
## Ethical Considerations
The Census Income dataset contains demographic attributes such as race and sex. Historical and social inequalities represented in the data may introduce bias into the model.Model performance may also vary across demographic groups. For this reason, predictions should not be interpreted as fair or unbiased simply because the overall model metrics are acceptable. The slice-based evaluation is used to help identify differences in performance across groups.
## Caveats and Recommendations
The model is trained on historical Census data and may not represent current population or income patterns. Some categorical groups contain relatively few observations,which can produce unstable slice metrics. For example, very small slices may report perfect precision, recall, and F1 scores that should not be interpreted as evidence of perfect generalizaton. Future work should evaluate fairness across demographic groups, validate the model on newer data, investigate class imbalance, and compare additional classification algorithms.
