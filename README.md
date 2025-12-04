# Final Report

Design and develop a shell interface that supports input/output redirection, piping, background processing, and a set of built-in functions. 

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

```bash
make
```
This will build the executable in ...
### Execution
```bash
make run
```
This will run the program ...

