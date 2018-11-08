Housing Prices
==============

Course Project  
CS771A: Introduction to Machine Learning

Team
----

Aayushi Pandey  
Ankita Jain  
Anshul Goel  
Hemanth Bollamreddi  
Navanya Sharma  
Pawan Harendra Mishra

Collaboration
------------

1. Clone and enter into the repository  
```
git clone https://github.com/anshul96go/CS771_Housing_Prices.git
cd path/to/cloned-directory
```

2. Contributers should make changes and test their code in their specific branch  
Each branch should be named as *user-initials* followed by *working-branch*  
Example: for initials *XYZabc* typing `git checkout -b xyzabc-working-branch` will create and checkout into the *xyzabc-working-branch*  
Omit the `-b` flag if the branch already exists  
Do not make a new branch each time some code is to be added or modified

3. Merge the branches post tests

Goal
----

Come up with a suitable technique to predict the sales price of a house given various attributes  
Use various machine learning models like support vector machines, regression, non-linear kernalized models and so on to get robust estimates of the housing prices

Data
-----
The Raw Data is available [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)  
Results and Outputs can be found [here](https://drive.google.com/drive/folders/1W2xhJBTJ_nndlCDLJPFl-mL6uXBBLkCw?usp=sharing)

Add links here

Code
----

### Data Cleaning

See file **data_cleaning.py**  
Edit the filenames (with location) of the train and test raw data files, in the header  
The program replaces missing data of certain features with 0, wheres drops the entire data point for certain other features. Categorical (nominal and ordinal data) are one hot encoded (converted to dummy/indicator variables)  
The program saves the cleaned data in the current directory  
The program also computes the correlation of features with dependant variable using the training data and saves the result in the current directory  


#### Requirement

* python 3.6 or higher
* pandas

### Handling Missing Data

Will be updated later

Update code details here. Remove template message post completion of project.
