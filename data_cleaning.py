import pandas as pd
from pandas.api.types import CategoricalDtype

# location of data files
# verify locations before running the code
train_file = "../data/train.csv"
test_file = "../data/test.csv"

# variables' definition
dependant_variable = 'SalePrice'
integral_variables = ['LotFrontage','LotArea','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','SalePrice']
time_variables = ['YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold']
nominal_variables_values = {'MSSubClass':['20','30','40','50','60','70','75','80','85','90','120','150','160','180','190'],
                    'MSZoning':['A','C','FV','I','RH','RL','RP','RM'],
                    'Street':['Grvl','Pave'],
                    'Alley':['Grvl','Pave','NA'],
                    'LotShape':['Reg','IR1','IR2','IR3'],
                    'LandContour':['Lvl','Bnk','HLS','Low'],
                    'Utilities':['AllPub','NoSewr','NoSeWa','ELO'],
                    'LotConfig':['Inside','Corner','CulDSac','FR2','FR3'],
                    'LandSlope':['Gtl','Mod','Sev'],
                    'Neighborhood':['CollgCr','Veenker','Crawfor','NoRidge','Mitchel','Somerst','NWAmes','OldTown','BrkSide','Sawyer','NridgHt','NAmes','SawyerW','IDOTRR','MeadowV','Edwards','Timber','Gilbert','StoneBr','ClearCr','NPkVill','Blmngtn','BrDale','SWISU','Blueste'],
                    'Condition1':['Norm','Feedr','PosN','Artery','RRAe','RRNn','RRAn','PosA','RRNe'],
                    'Condition2':['Norm','Feedr','PosN','Artery','RRAe','RRNn','RRAn','PosA','RRNe'],
                    'BldgType':['1Fam','2fmCon','Duplex','TwnhsE','Twnhs'],
                    'HouseStyle':['2Story','1Story','1.5Fin','1.5Unf','SFoyer','SLvl','2.5Unf','2.5Fin'],
                    'RoofStyle':['Gable','Hip','Gambrel','Mansard','Flat','Shed'],
                    'RoofMatl':['CompShg','WdShngl','Metal','WdShake','Membran','Tar&Grv','Roll','ClyTile'],
                    'Exterior1st':['VinylSd','MetalSd','Wd Sdng','HdBoard','BrkFace','WdShing','CemntBd','Plywood','AsbShng','Stucco','BrkComm','AsphShn','Stone','ImStucc','CBlock','Other','PreCast'],
                    'Exterior2nd':['VinylSd','MetalSd','Wd Sdng','HdBoard','BrkFace','WdShing','CemntBd','Plywood','AsbShng','Stucco','BrkComm','AsphShn','Stone','ImStucc','CBlock','Other','PreCast'],
                    'MasVnrType':['BrkFace','CBlock','Stone','BrkCmn','None'],
                    'ExterQual':['Gd','TA','Ex','Fa','Po'],
                    'ExterCond':['Gd','TA','Ex','Fa','Po'],
                    'Foundation':['PConc','CBlock','BrkTil','Wood','Slab','Stone'],
                    'BsmtQual':['Ex','Gd','TA','Fa','Po','NA'],
                    'BsmtCond':['Ex','Gd','TA','Fa','Po','NA'],
                    'BsmtExposure':['No','Gd','Mn','Av','NA'],
                    'BsmtFinType1':['GLQ','ALQ','Unf','Rec','BLQ','NA','LwQ'],
                    'BsmtFinType2':['GLQ','ALQ','Unf','Rec','BLQ','NA','LwQ'],
                    'Heating':['GasA','GasW','Grav','Wall','OthW','Floor'],
                    'HeatingQC':['Ex','Gd','TA','Fa','Po'],
                    'CentralAir':['N','Y'],
                    'Electrical':['SBrkr','FuseF','FuseA','FuseP','Mix'],
                    'KitchenQual':['Gd','TA','Ex','Fa','Po'],
                    'Functional':['Typ','Min1','Maj1','Min2','Mod','Maj2','Sev','Sal'],
                    'FireplaceQu':['Ex','Gd','TA','Fa','Po','NA'],
                    'GarageType':['Attchd','Detchd','BuiltIn','CarPort','NA','Basment','2Types'],
                    'GarageFinish':['RFn','Unf','Fin','NA'],
                    'GarageQual':['Ex','Gd','TA','Fa','Po','NA'],
                    'GarageCond':['Ex','Gd','TA','Fa','Po','NA'],
                    'PavedDrive':['Y','P','N'],
                    'PoolQC':['NA','Ex','Fa','Gd','TA'],
                    'Fence':['NA','MnPrv','GdWo','GdPrv','MnWw'],
                    'MiscFeature':['NA','Shed','Gar2','Othr','TenC','Elev'],
                    'SaleType':['WD','New','COD','ConLD','ConLI','CWD','ConLw','Con','Oth','VMD'],
                    'SaleCondition':['Normal','Abnorml','Partial','AdjLand','Alloca','Family']}

nominal_variables = []
nominal_variables_without_NA = []
for var in nominal_variables_values:
    nominal_variables.append(var)
    if 'NA' not in nominal_variables_values[var]: nominal_variables_without_NA.append(var)

def clean_train_data():
    df1 = pd.read_csv(train_file, index_col='Id', keep_default_na=False)
    df2 = df1[integral_variables].replace('NA', 0).astype(int)
    df3 = df1[nominal_variables].astype(str)
    # drop data points having NA for features not supporting NA
    for var in nominal_variables_without_NA:
        df3 = df3[df3[var] != 'NA']
    # define feature categories for all nominal/ordinal features
    for var in nominal_variables:
        cat_dtype = CategoricalDtype(categories=nominal_variables_values[var])
        df3[var] = df3[var].astype(cat_dtype)
    # one hot encoding of categorical variables
    ohe_df3 = pd.get_dummies(df3, prefix=nominal_variables, columns=nominal_variables)
    df = pd.merge(df2, ohe_df3, how='inner', right_index=True, left_index=True)
    # output dropout and total data
    total_data = df1.shape[0]
    clean_data = df.shape[0]
    print("training data cleaned with dropout of", total_data - clean_data)
    print("total clean train data", clean_data)
    # save clean training data
    df.to_csv("clean_train.csv")
    return df

def clean_test_data():
    df1 = pd.read_csv(test_file, index_col='Id', keep_default_na=False)
    df2 = df1[integral_variables[:-1]].replace('NA', 0).astype(int)
    df3 = df1[nominal_variables].astype(str)
    # drop data points having NA for features not supporting NA
    for var in nominal_variables_without_NA:
        df3 = df3[df3[var] != 'NA']
    # define feature categories for all nominal/ordinal features
    for var in nominal_variables:
        cat_dtype = CategoricalDtype(categories=nominal_variables_values[var])
        df3[var] = df3[var].astype(cat_dtype)
    # one hot encoding of categorical variables
    ohe_df3 = pd.get_dummies(df3, prefix=nominal_variables, columns=nominal_variables)
    df = pd.merge(df2, ohe_df3, how='inner', right_index=True, left_index=True)
    # output dropout and total data
    total_data = df1.shape[0]
    clean_data = df.shape[0]
    print("testing data cleaned with dropout of", total_data - clean_data)
    print("total clean test data", clean_data)
    # save clean testing data
    df.to_csv("clean_test.csv")
    return df

train_df = clean_train_data()
test_df = clean_test_data()