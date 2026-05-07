.. code:: ipython3

    import seaborn as sns
    import matplotlib.pyplot as plt

.. code:: ipython3

    titanic = sns.load_dataset('titanic')
    
    print(titanic.head())


.. parsed-literal::

       survived  pclass     sex   age  sibsp  parch     fare embarked  class  \
    0         0       3    male  22.0      1      0   7.2500        S  Third   
    1         1       1  female  38.0      1      0  71.2833        C  First   
    2         1       3  female  26.0      0      0   7.9250        S  Third   
    3         1       1  female  35.0      1      0  53.1000        S  First   
    4         0       3    male  35.0      0      0   8.0500        S  Third   
    
         who  adult_male deck  embark_town alive  alone  
    0    man        True  NaN  Southampton    no  False  
    1  woman       False    C    Cherbourg   yes  False  
    2  woman       False  NaN  Southampton   yes   True  
    3  woman       False    C  Southampton   yes  False  
    4    man        True  NaN  Southampton    no   True  
    

.. code:: ipython3

    # Display Survival Count Table
    survival_table = titanic.pivot_table(
        index='sex',
        columns='survived',
        aggfunc='size',
        fill_value=0
    )
    
    print("Survival Table:\n")
    print(survival_table)


.. parsed-literal::

    Survival Table:
    
    survived    0    1
    sex               
    female     81  233
    male      468  109
    

.. code:: ipython3

    # Box Plot
    plt.figure(figsize=(8,5))
    
    sns.boxplot(
        x='sex',
        y='age',
        hue='survived',
        data=titanic
    )
    
    plt.title("Age Distribution by Gender and Survival")
    plt.xlabel("Gender")
    plt.ylabel("Age")
    
    plt.show()



.. image:: output_3_0.png


.. code:: ipython3

    # Violin Plot
    plt.figure(figsize=(8,5))
    
    sns.violinplot(
        x='sex',
        y='age',
        hue='survived',
        data=titanic,
        split=True
    )
    
    plt.title("Violin Plot of Age by Gender and Survival")
    plt.xlabel("Gender")
    plt.ylabel("Age")
    
    plt.show()



.. image:: output_4_0.png



