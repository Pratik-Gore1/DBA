.. code:: ipython3

    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

.. code:: ipython3

    # Dataset Information
    
    print("Dataset Name : Titanic Dataset")
    print("Source URL : https://www.kaggle.com/datasets/yasserh/titanic-dataset")
    print("Description : The dataset contains information about passengers such as age, gender, ticket fare, class, and survival status.")


.. parsed-literal::

    Dataset Name : Titanic Dataset
    Source URL : https://www.kaggle.com/datasets/yasserh/titanic-dataset
    Description : The dataset contains information about passengers such as age, gender, ticket fare, class, and survival status.
    

.. code:: ipython3

    df = pd.read_csv("data.csv")
    
    print(df.head())


.. parsed-literal::

       PassengerId  Survived  Pclass  \
    0            1         0       3   
    1            2         1       1   
    2            3         1       3   
    3            4         1       1   
    4            5         0       3   
    
                                                    Name     Sex   Age  SibSp  \
    0                            Braund, Mr. Owen Harris    male  22.0      1   
    1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
    2                             Heikkinen, Miss. Laina  female  26.0      0   
    3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
    4                           Allen, Mr. William Henry    male  35.0      0   
    
       Parch            Ticket     Fare Cabin Embarked  
    0      0         A/5 21171   7.2500   NaN        S  
    1      0          PC 17599  71.2833   C85        C  
    2      0  STON/O2. 3101282   7.9250   NaN        S  
    3      0            113803  53.1000  C123        S  
    4      0            373450   8.0500   NaN        S  
    

.. code:: ipython3

    print("Number of Rows and Columns :", df.shape)


.. parsed-literal::

    Number of Rows and Columns : (891, 12)
    

.. code:: ipython3

    print(df.dtypes)


.. parsed-literal::

    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object
    

.. code:: ipython3

    # Check Missing Values
    print(df.isnull().sum())


.. parsed-literal::

    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64
    

.. code:: ipython3

    df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    df = df.drop('Cabin', axis=1)

.. code:: ipython3

    print(df.isnull().sum())


.. parsed-literal::

    PassengerId    0
    Survived       0
    Pclass         0
    Name           0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Embarked       0
    dtype: int64
    

.. code:: ipython3

    # Statistical Information
    
    print(df.describe())


.. parsed-literal::

           PassengerId    Survived      Pclass         Age       SibSp  \
    count   891.000000  891.000000  891.000000  891.000000  891.000000   
    mean    446.000000    0.383838    2.308642   29.699118    0.523008   
    std     257.353842    0.486592    0.836071   13.002015    1.102743   
    min       1.000000    0.000000    1.000000    0.420000    0.000000   
    25%     223.500000    0.000000    2.000000   22.000000    0.000000   
    50%     446.000000    0.000000    3.000000   29.699118    0.000000   
    75%     668.500000    1.000000    3.000000   35.000000    1.000000   
    max     891.000000    1.000000    3.000000   80.000000    8.000000   
    
                Parch        Fare  
    count  891.000000  891.000000  
    mean     0.381594   32.204208  
    std      0.806057   49.693429  
    min      0.000000    0.000000  
    25%      0.000000    7.910400  
    50%      0.000000   14.454200  
    75%      0.000000   31.000000  
    max      6.000000  512.329200  
    

.. code:: ipython3

    categorical_columns = ['Sex', 'Embarked']
    
    print("Categorical Columns :")
    print(categorical_columns)


.. parsed-literal::

    Categorical Columns :
    ['Sex', 'Embarked']
    

.. code:: ipython3

    # Convert Categorical Variables into Numerical Variables
    
    label_encoder = LabelEncoder()
    
    for column in categorical_columns:
        df[column] = df[column].astype(str)
        df[column] = label_encoder.fit_transform(df[column])
    
    print(df.head())


.. parsed-literal::

       PassengerId  Survived  Pclass  \
    0            1         0       3   
    1            2         1       1   
    2            3         1       3   
    3            4         1       1   
    4            5         0       3   
    
                                                    Name  Sex   Age  SibSp  Parch  \
    0                            Braund, Mr. Owen Harris    1  22.0      1      0   
    1  Cumings, Mrs. John Bradley (Florence Briggs Th...    0  38.0      1      0   
    2                             Heikkinen, Miss. Laina    0  26.0      0      0   
    3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    0  35.0      1      0   
    4                           Allen, Mr. William Henry    1  35.0      0      0   
    
                 Ticket     Fare  Embarked  
    0         A/5 21171   7.2500         2  
    1          PC 17599  71.2833         0  
    2  STON/O2. 3101282   7.9250         2  
    3            113803  53.1000         2  
    4            373450   8.0500         2  
    

.. code:: ipython3

    '''print("""
    Variable Descriptions:
    PassengerId : Unique ID of passenger
    Survived    : Survival status (0 = No, 1 = Yes)
    Pclass      : Passenger class
    Name        : Name of passenger
    Sex         : Gender of passenger
    Age         : Age in years
    SibSp       : Number of siblings/spouses aboard
    Parch       : Number of parents/children aboard
    Ticket      : Ticket number
    Fare        : Passenger fare
    Cabin       : Cabin number
    Embarked    : Port of embarkation""")'''

