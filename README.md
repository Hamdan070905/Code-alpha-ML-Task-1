 Code-alpha-ML-Task-1

1. Data Generation / Loading
   
           > A synthetic dataset is created with financial features such as:
                 > income, debt_ratio, credit_utilization, age, etc.
           > The target column (default) indicates if a person will default (1) or not (0).

2. Data Preprocessing

           > Split the data into features (X) and target (y).
           > Perform train-test split (80% training, 20% testing).
           > Use StandardScaler to normalize the feature values for better model performance.

3. Model Training

       > Trained three classification models:
             >  Logistic Regression – linear model for binary classification
             >  Decision Tree – splits data based on conditions
             > Random Forest – ensemble of decision trees for better accuracy
       > Each model is trained on the training set and used to predict on the test set.

4. Model Evaluation

       For each model:
             > Printed the confusion matrix and classification report showing:
             > Precision, Recall, F1-Score, and Accuracy
             > Calculated the ROC AUC Score to measure how well the model distinguishes between defaulters and non-defaulters.

5. ROC Curve Plotting

           > Plotted the ROC (Receiver Operating Characteristic) curves for all models to compare their performance.
           > This helps visualize the trade-off between True Positive Rate and False Positive Rate.
           > Included the AUC value (Area Under Curve) on the plot for each model.


