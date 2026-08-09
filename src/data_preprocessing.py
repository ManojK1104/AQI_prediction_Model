from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
def process(df):

    df.drop_duplicates(inplace=True)

    X=df.drop(columns=['Date','Time','RH'])
    y=df['RH']

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

    categorical_column=X.select_dtypes(include=object).columns
    numerical_columns=X.select_dtypes(exclude=object).columns

    numerical_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                        ('scaler',MinMaxScaler())])

    categorical_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                            ('encoder',OneHotEncoder(
                                                drop='first',handle_unknown='ignore',
                                                                    sparse_output=False))
                                                                    ])

    transformer=ColumnTransformer(transformers=[("Cat",categorical_pipeline,categorical_column),
                                        ("num",numerical_pipeline,numerical_columns)])

    X_train=transformer.fit_transform(X_train)
    X_test=transformer.transform(X_test)

    return X_train,X_test,y_train,y_test,transformer

