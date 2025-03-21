# Generating nice smooth parallel plots!

<img src="https://raw.githubusercontent.com/markste-in/parallelplot/refs/heads/main/README_files/output_9_0.png" height="170" />
&nbsp;
<img src="https://raw.githubusercontent.com/markste-in/parallelplot/refs/heads/main/README_files/output_13_0.png" height="170" />

## How to install

Just run

```shell
pip install parallelplot
```

## Little Demo on the Wine Quality Dataset

### First lets import some packages we need to get some sample data


```python
# Import libraries to handle data 
import numpy as np
import pandas as pd

# The only thing that is really needs to be imported 
# is the plot function from the parallelplot module 
# and the pyplot module from matplotlib to display the plot
import parallelplot.plot as pp
import matplotlib.pyplot as plt


# There is also a module that contains a nice colormap. In addition you can use the matplotlib colormap module
from parallelplot.cmaps import purple_blue
import matplotlib.cm as cm

```


```python
# Function to download and load the wine quality dataset
def load_wine_quality_dataset():
    # URLs for the Wine Quality datasets 
    red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    # Download and read the datasets
    red_wine = pd.read_csv(red_wine_url, sep=';')
    white_wine = pd.read_csv(white_wine_url, sep=';')
    
    # Add a wine type column
    red_wine['wine_type'] = 'red'
    white_wine['wine_type'] = 'white'
    
    # Combine the datasets
    wine_df = pd.concat([red_wine, white_wine], axis=0, ignore_index=True)
    
    return wine_df


wine_df = load_wine_quality_dataset()

```


```python
print("Wine Quality Dataset:")
wine_df
```

    Wine Quality Dataset:





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>wine_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.99680</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.99700</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.99800</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.99780</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6492</th>
      <td>6.2</td>
      <td>0.21</td>
      <td>0.29</td>
      <td>1.6</td>
      <td>0.039</td>
      <td>24.0</td>
      <td>92.0</td>
      <td>0.99114</td>
      <td>3.27</td>
      <td>0.50</td>
      <td>11.2</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>6493</th>
      <td>6.6</td>
      <td>0.32</td>
      <td>0.36</td>
      <td>8.0</td>
      <td>0.047</td>
      <td>57.0</td>
      <td>168.0</td>
      <td>0.99490</td>
      <td>3.15</td>
      <td>0.46</td>
      <td>9.6</td>
      <td>5</td>
      <td>white</td>
    </tr>
    <tr>
      <th>6494</th>
      <td>6.5</td>
      <td>0.24</td>
      <td>0.19</td>
      <td>1.2</td>
      <td>0.041</td>
      <td>30.0</td>
      <td>111.0</td>
      <td>0.99254</td>
      <td>2.99</td>
      <td>0.46</td>
      <td>9.4</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>6495</th>
      <td>5.5</td>
      <td>0.29</td>
      <td>0.30</td>
      <td>1.1</td>
      <td>0.022</td>
      <td>20.0</td>
      <td>110.0</td>
      <td>0.98869</td>
      <td>3.34</td>
      <td>0.38</td>
      <td>12.8</td>
      <td>7</td>
      <td>white</td>
    </tr>
    <tr>
      <th>6496</th>
      <td>6.0</td>
      <td>0.21</td>
      <td>0.38</td>
      <td>0.8</td>
      <td>0.020</td>
      <td>22.0</td>
      <td>98.0</td>
      <td>0.98941</td>
      <td>3.26</td>
      <td>0.32</td>
      <td>11.8</td>
      <td>6</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
<p>6497 rows × 13 columns</p>
</div>




```python
# Manipulate the dataset to simulate small and large numbers
wine_df["fixed acidity"] = wine_df["fixed acidity"] * 1e6
wine_df["volatile acidity"] = wine_df["volatile acidity"] / 1e6
```

## Create the plots from the imported data!


```python
# Example 1: Basic parallel plot with default style
fig1, axes1 = pp.plot(
    df=wine_df,
    target_column='quality',
    title="Wine Quality Dataset - All Features",
    figsize=(16, 8),
    tick_label_size=10,
    alpha=0.3,
    cmap=cm.hot,
    order='max',
    lw=0.5,
    
)
plt.show()

```


    
![png](https://raw.githubusercontent.com/markste-in/parallelplot/refs/heads/main/README_files/output_8_0.png)
    



```python
# Example 2: Parallel plot with dark background
fig2, axes2 = pp.plot(
    df=wine_df,
    target_column='quality',
    title="Wine Quality Dataset - Dark Background",
    figsize=(16, 8),
    style="dark_background",
    lw=0.2,
    # axes_to_reverse = [0, 1, 2, 5]
)
plt.show()
```


    
![png](https://raw.githubusercontent.com/markste-in/parallelplot/refs/heads/main/README_files/output_9_0.png)
    



```python
# Example 3: Different cmap 
fig3, axes3 = pp.plot(
    df=wine_df,
    target_column='quality',
    title="Wine Quality Dataset - Colored by Wine Type",
    figsize=(16, 8),
    cmap=purple_blue,
    style="dark_background",
    lw=0.1,
    order='min',
    alpha = 0.2,
    axes_to_reverse = [1,2]
)
plt.show()

```


    
![png](https://raw.githubusercontent.com/markste-in/parallelplot/refs/heads/main/README_files/output_10_0.png)
    



```python
# Example 4: Select top features with highest correlation to quality
# Calculate correlations with quality
corr_with_quality = wine_df.drop(columns=['wine_type']).corr()['quality'].abs().sort_values(ascending=False)
top_features = corr_with_quality.index[:8]  # Top 8 features

# Create subset with only the top features
wine_top_features = wine_df[top_features]

fig4, axes4 = pp.plot(
    df=wine_top_features,
    target_column='quality',
    title="Wine Quality - Top Correlated Features",
    figsize=(14, 7),
    cmap=cm.viridis,
    style="dark_background",
    lw=0.2,
    axes_to_reverse = [1,2]


)
plt.show()

```


    
![png](https://raw.githubusercontent.com/markste-in/parallelplot/refs/heads/main/README_files/output_11_0.png)
    



```python
# Example 3: Different cmap 
fig3, axes3 = pp.plot(
    df=wine_df,
    target_column='quality',
    title="Wine Quality Dataset - Colored by Wine Type",
    figsize=(16, 8),
    cmap=cm.plasma,
    style="dark_background",
    lw=0.1,
    axes_to_reverse = [1,2]

)
plt.show()
```


    
![png](https://raw.githubusercontent.com/markste-in/parallelplot/refs/heads/main/README_files/output_12_0.png)
    



```python
# Example 3: Different cmap and hide all axes
fig3, axes3 = pp.plot(
    df=wine_df,
    target_column='quality',
    title="Wine Quality Dataset - Colored by Wine Type",
    figsize=(16, 8),
    cmap=cm.cool.reversed(),
    style="dark_background",
    lw=0.1,
    # order='random',
    hide_axes=True,
    axes_to_reverse = [0]


)
plt.show()
```


    
![png](https://raw.githubusercontent.com/markste-in/parallelplot/refs/heads/main/README_files/output_13_0.png)
    

