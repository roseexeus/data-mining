# Final Report

Goal: Using machine learning techniques to understand the drivers of decisions made on a daily basis.

## Group Members
- **Rose Exeus**: re22c@fsu.edu
- **Chris Cargill**: cac19r@fsu.edu

  
## File Listing 
```
images/
│ ├── lr_satisfaction_analysis.png
│ ├── nb_regret_analysis.png
│ ├── nb_satisfaction_analysis.png
│ └── lr_regret_analysis.png
│
├── final_analysis.py
├── README.md
└── transformed_dataset.csv
```
## Running the program

### Required Libraries
- **Itertools**: from this, will need to import **Combinations**
- **Matplotlib**
- **Numpy**
- **Os**
- **Pandas**
- **Pickle**
- **Seaborn**
- **Sklearn.ensemble**: from this, will need to import **RandomForestRegressor**
- **Sklearn.linear_model**: from this, will need to import **LogisticRegression**
- **Sklearn.metrics**: from this, will need to import *classification_report*, *confusion_matrix*, *mean_squared_error*, *r2_score*, and *mean_absolute_error*
- **Sklearn.model_selection**: from this, will need to import *train_test_split* and *cross_val_score*
- **Sklearn.naive_bayes**: from this, will need to import **GaussianNB**

Our fully data set is stored under transformed_dataset.csv, and it has been pre-loaded in the python code file.
To run the program...
### Execution
```bash
python final_analysis.py
```
Or...
```bash
py final_analysis.py
```
This will display the results/metrics of our trained models in the terminal. It will also create an images folder and produce 4 different images, each showing the visualizations of our training and testing, along with the results of the study. 

Execution time: Less than 1 minute, approximately 28.9 seconds. 

