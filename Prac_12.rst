.. code:: ipython3

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

.. code:: ipython3

    iris = pd.read_csv("iris.csv")
    
    # Remove Id column if present
    if 'Id' in iris.columns:
        iris = iris.drop('Id', axis=1)
        
    print("First 5 Rows of Dataset:\n")
    print(iris.head())


.. parsed-literal::

    First 5 Rows of Dataset:
    
       SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      species
    0            5.1           3.5            1.4           0.2  Iris-setosa
    1            4.9           3.0            1.4           0.2  Iris-setosa
    2            4.7           3.2            1.3           0.2  Iris-setosa
    3            4.6           3.1            1.5           0.2  Iris-setosa
    4            5.0           3.6            1.4           0.2  Iris-setosa
    

.. code:: ipython3

    # 1. Features and Their Types
    print("\nFeatures and Data Types:\n")
    
    for column in iris.columns:
    
        if iris[column].dtype == 'object':
            feature_type = "Nominal"
    
        else:
            feature_type = "Numeric"
    
        print(column, ":", feature_type)


.. parsed-literal::

    
    Features and Data Types:
    
    SepalLengthCm : Numeric
    SepalWidthCm : Numeric
    PetalLengthCm : Numeric
    PetalWidthCm : Numeric
    species : Nominal
    

.. code:: ipython3

    # 2. Histogram for Each Feature
    # 3. Demonstration of Subplots
    numeric_features = iris.select_dtypes(include='number').columns
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes = axes.ravel()
    
    for i, column in enumerate(numeric_features):
    
        axes[i].hist(iris[column], bins=15)
    
        axes[i].set_title(column)
    
        axes[i].set_xlabel("Value")
    
        axes[i].set_ylabel("Frequency")
    
    plt.tight_layout()
    
    plt.show()



.. image:: output_3_0.png


.. code:: ipython3

    # 4. Boxplot for Each Feature
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes = axes.ravel()
    
    for i, column in enumerate(numeric_features):
    
        sns.boxplot(y=iris[column], ax=axes[i])
    
        axes[i].set_title(column)
    
    plt.tight_layout()
    
    plt.show()



.. image:: output_4_0.png


