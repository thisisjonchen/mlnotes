# Machine Learning Notes 📝

## Table of Contents
1. [Tools](#tools)
2. [What is Machine Learning?](#what-is-machine-learning)
3. [Supervised Learning](#supervised-learning)
    * 3.1 [Linear Regression](#linear-regression)
5. [Unsupervised Learning](#unsupervised-learning)

# Tools
- Jupyter Notebook
- Libraries
  - NumPy, scientific computing
  - Matplotlib, plotting data


# What is Machine Learning?
Arthur Samuel, a pioneer in CS + AI in his time, defined machine learning as "...[the] field of study that gives computers the ability to learn without being explicitly programmed" (1959). He helped evolve AI by writing the first checkers-playing game that learned from thousands of games against itself.

Machine Learning Algorithms:
- Supervised Learning (*Used in most real-world applications/rapid advancements)
- Unsupervised Learning
- Recommender Systems
- Reinforcement Learning


# Supervised Learning
`(DEF)` **Supervised Learning**: Learning via input (x) to output (y) mappings.\
Key characteristic: **Learns from being given examples and the "right answers"**

Given input(s), the algorithm will make a "guess" and output based on what it had previously learned.

Some examples Andrew provided:
| Input (X) | Output (Y)  | Application  |
|---|---|---|
|  email | spam? (0/1)  | spam filtering  |
| audio  |  text transcripts |  speech recognition |
|  ad, user info | click? (0/1)  |  online advertising |
| image, radar info | position of other cars | self-driving car |

### Major Types of Supervised Learning
`(DEF)` **Regression**: Predict a number from infinitely many possible outputs
- Ex: Housing Market Crisis (Sq. Feet to Market Value)
- Graphing technique utilizes a best-fitting line (linear, logarithmic, etc.)
- Models: Linear Regression
  
`(DEF)` **Classification**: Predict categories from a small number of possible outputs
- Ex: Breast Cancer Detection (Benign vs. Not-Benign)
- Terminology: Classes/Categories are often used interchangeably with the output
- Graphing technique utilizes a boundary line depending on input(s), separating one class/category from another


## Linear Regression
`(DEF)` **Linear Regression**: Fits a best-fitting, straight (linear) line to your data

`(DEF)` **Training Set**: Data used to train the model
- Notation:
    - $x$ = features, "input" variable
    - $y$ = targets, "output" variable
    - $m$ = number of training examples
    - $f$ = model after training wherein we plug in $x$ to get $\hat{y}$
    - $\hat{y}$ = prediction for y
    - $(x, y)$ = single training example (pair)
    - $(x^{(i)}, y^{(i)})$ = $i^{th}$ training example with relation to the $i^th$ row (1st, 2nd, 3rd...)
      - *NOTE*: $x^{(i)}$ is not exponentiation, but denotes the row
      - $(x^{(1)}, y^{(1)})$ refers to the 1st training example at row 1 of the training set
     
Process:
- Training Set &#8594; Learning Algorithm &#8594; Model $f$
- $x$ &#8594; Model $f$ &#8594; $\hat{y}$
    - Ex: size &#8594; Model $f$ &#8594; estimated price

**How to represent $f$**: $f_{w,b}(x)=wx+b$\
*Can also drop the subscript $w,b$ &#8594; $f(x)=wx+b$
- Notation:
  - $w$ = parameter: weight
  - $b$ = parameter: bias

`(DEF)` **Univariate Linear Regression**: Fancy name for linear regression with one variable (single feature x)






# Unsupervised Learning
`(DEF)` **Unsupervised Learning**: Learning and structuring from data that only comes with inputs (x), but not output labels (y).\
Key characteristic: **Finds something interesting (patterns, structures, clusters, etc.) in unlabeled data -- we don't tell it what's right and wrong**

### Major Types of Unsupervised Learning
`(DEF)` **Clustering**: Groups similar data points together
- Ex: Google News (matching headlines into a recommendation list)

`(DEF)` **Anomaly Detection**: Finds unusual data points

`(DEF)` **Dimensionality Reduction**: Compress data using fewer numbers


