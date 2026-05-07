.. code:: ipython3

    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

.. code:: ipython3

    df = pd.read_csv("HousingData.csv")
    
    print(df.head())


.. parsed-literal::

          CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  TAX  PTRATIO  \
    0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900    1  296     15.3   
    1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671    2  242     17.8   
    2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671    2  242     17.8   
    3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622    3  222     18.7   
    4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622    3  222     18.7   
    
            B  LSTAT  MEDV  
    0  396.90   4.98  24.0  
    1  396.90   9.14  21.6  
    2  392.83   4.03  34.7  
    3  394.63   2.94  33.4  
    4  396.90    NaN  36.2  
    

.. code:: ipython3

    # Shape of Dataset
    
    print("Shape of Dataset :", df.shape)
    
    # Column Names
    
    print("\nColumns in Dataset :\n")
    
    print(df.columns)
    
    # Check Null Values
    
    print("\nNull Values :\n")
    
    print(df.isnull().sum())


.. parsed-literal::

    Shape of Dataset : (506, 14)
    
    Columns in Dataset :
    
    Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
           'PTRATIO', 'B', 'LSTAT', 'MEDV'],
          dtype='object')
    
    Null Values :
    
    CRIM       20
    ZN         20
    INDUS      20
    CHAS       20
    NOX         0
    RM          0
    AGE        20
    DIS         0
    RAD         0
    TAX         0
    PTRATIO     0
    B           0
    LSTAT      20
    MEDV        0
    dtype: int64
    

.. code:: ipython3

    # Fill Missing Values with Mean
    
    df = df.fillna(df.mean(numeric_only=True))
    
    # Verify Again
    
    print("\nMissing Values After Filling:\n")
    
    print(df.isnull().sum())


.. parsed-literal::

    
    Missing Values After Filling:
    
    CRIM       0
    ZN         0
    INDUS      0
    CHAS       0
    NOX        0
    RM         0
    AGE        0
    DIS        0
    RAD        0
    TAX        0
    PTRATIO    0
    B          0
    LSTAT      0
    MEDV       0
    dtype: int64
    

.. code:: ipython3

    # Independent Variables
    
    X = df.drop('MEDV', axis=1)
    
    # Dependent Variable
    
    y = df['MEDV']
    
    print(X.head())
    
    print(y.head())


.. parsed-literal::

          CRIM    ZN  INDUS  CHAS    NOX     RM   AGE     DIS  RAD  TAX  PTRATIO  \
    0  0.00632  18.0   2.31   0.0  0.538  6.575  65.2  4.0900    1  296     15.3   
    1  0.02731   0.0   7.07   0.0  0.469  6.421  78.9  4.9671    2  242     17.8   
    2  0.02729   0.0   7.07   0.0  0.469  7.185  61.1  4.9671    2  242     17.8   
    3  0.03237   0.0   2.18   0.0  0.458  6.998  45.8  6.0622    3  222     18.7   
    4  0.06905   0.0   2.18   0.0  0.458  7.147  54.2  6.0622    3  222     18.7   
    
            B      LSTAT  
    0  396.90   4.980000  
    1  396.90   9.140000  
    2  392.83   4.030000  
    3  394.63   2.940000  
    4  396.90  12.715432  
    0    24.0
    1    21.6
    2    34.7
    3    33.4
    4    36.2
    Name: MEDV, dtype: float64
    

.. code:: ipython3

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )
    
    print("Training Data Shape :", X_train.shape)
    
    print("Testing Data Shape :", X_test.shape)


.. parsed-literal::

    Training Data Shape : (379, 13)
    Testing Data Shape : (127, 13)
    

.. code:: ipython3

    # Create Model
    
    model = LinearRegression()
    
    # Train Model
    
    model.fit(X_train, y_train)
    
    print("Model Training Completed")


.. parsed-literal::

    Model Training Completed
    

.. code:: ipython3

    # Predict Output
    
    y_pred = model.predict(X_test)
    
    print("Predicted Prices :\n")
    
    print(y_pred[:10])


.. parsed-literal::

    Predicted Prices :
    
    [29.1000167  36.50596936 14.75387304 25.3326485  18.53212239 22.99322946
     18.13561219 14.57509445 22.12297534 20.86176246]
    

.. code:: ipython3

    # Calculate Mean Squared Error
    
    mse = mean_squared_error(y_test, y_pred)
    
    print("Mean Squared Error (MSE) :", mse)


.. parsed-literal::

    Mean Squared Error (MSE) : 22.406361917537282
    

.. code:: ipython3

    comparison = pd.DataFrame({
        'Actual Price': y_test.values,
        'Predicted Price': y_pred
    })
    
    print(comparison.head(10))


.. parsed-literal::

       Actual Price  Predicted Price
    0          23.6        29.100017
    1          32.4        36.505969
    2          13.6        14.753873
    3          22.8        25.332648
    4          16.1        18.532122
    5          20.0        22.993229
    6          17.8        18.135612
    7          14.0        14.575094
    8          19.6        22.122975
    9          16.8        20.861762
    

.. code:: ipython3

    plt.figure(figsize=(8,6))
    
    plt.scatter(y_test, y_pred)
    
    plt.xlabel("Actual Prices")
    
    plt.ylabel("Predicted Prices")
    
    plt.title("Actual vs Predicted House Prices")
    
    plt.show()



.. image:: output_10_0.png


