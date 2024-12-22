# Machine Learning Notes 📝

## Table of Contents
1. [Tools](#tools)
2. [What is Machine Learning?](#what-is-machine-learning)
3. [Supervised Learning](#supervised-learning)
    * 3.1 [Linear Regression](#linear-regression)
      * 3.11 [Cost Function](#cost-function)
      * 3.12 [Gradient Descent](#gradient-descent)
5. [Unsupervised Learning](#unsupervised-learning)

# Tools
- Language: Python
- Platform: Jupyter Notebook
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
- Ex: Housing Market Prices (Sq. Feet to Market Value)
- Graphing technique utilizes a best-fitting line (linear, logarithmic, etc.)
- Models: Linear Regression
  
`(DEF)` **Classification**: Predict categories from a small number of possible outputs
- Ex: Breast Cancer Detection (Benign vs. Not-Benign)
- Terminology: Classes/Categories are often used interchangeably with the output
- Graphing technique utilizes a boundary line depending on input(s), separating one class/category from another


## Linear Regression
`(DEF)` **Linear Regression**: Fits a best-fitting, straight (linear) line to your data

`(DEF)` **Univariate Linear Regression**: Fancy name for linear regression with one variable (single feature x)

`(DEF)` **Training Set**: Data used to train the model
- Notation:
    - $x$: features, "input" variable
    - $y$: targets, "output" variable
    - $m$: number of training examples
    - $f$: model, an equation obtained from training wherein we plug in $x$ to get $\hat{y}$
        - `(EQUATION)` $f_{w,b}(x)=wx+b$
        - Can also drop the subscript $w,b$ &#8594; $f(x)=wx+b$
           - $\hat{y} = w(x^{(i)}) + b$
           - $w$: parameter weight
           - $b$: parameter bias
    - $\hat{y}$: prediction for y
    - $(x, y)$: single training example (pair)
    - $(x^{(i)}, y^{(i)})$: $i^{th}$ training example with relation to the $i^th$ row (1st, 2nd, 3rd...)
      - *NOTE*: $x^{(i)}$ is not exponentiation, but denotes the row
      - $(x^{(1)}, y^{(1)})$ refers to the 1st training example at row 1 of the training set
     
Process:
- Training Set &#8594; Learning Algorithm &#8594; Model $f$
- $x$ &#8594; $f$ &#8594; $\hat{y}$
    - Ex: size &#8594; Model $f$ &#8594; estimated price

### Cost Function
Question: How to find how $\hat{y}$ compares to the true target $y^{(i)}$?\
Answer: Use a **cost function**

**Squared Error Cost Function**
- The most common used cost function for most regression related models
- `(EQUATION)` $J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2 = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$
- *NOTE*: Andrew says that the reason we divide by 2 here is to make future calculations "neater"

The squared error cost function is not the only cost function that exists -- there are more.

**The goal** of regression is to minimize the cost function $J(w,b)$
- When we use random $w$, we can get a graph with x-axis $w$ and y-axis $J(w)$ (note, this is excluding $b$ for now to make the example simpler). With this, we can find the minimum $J(w)$ and use it in our model $f$ (2D).
- With both $w, b$, we get a plot where it looks like one part of a hyperbolic paraboloid (or "hammock", "soup bowl", and "curved dinner plate"). This plot would have $b$ and $w$ as parameters/inputs on the bottom axes, and $J(w,b)$ on the vertical axis (3D).
  - This can also be accompanied by a contour (topographic) plot with $w$ on the x-axis, $b$ on the y-axis. At the center of the contour plot (where the lines are "growing" from) is where $J(w,b)$ is the minimum.

Now, how can we more easily find the minimum $w,b$? We can use an algorithm called **Gradient Descent**.

### Gradient Descent
One of the most important building blocks in machine learning, helps minimize some *any* function.

Outline:
- Start with some $w,b$ (a common approach is to first set $w$=0, $b$=0)
- Keep changing $w,b$ to reduce $J(w,b)$, until we settle at or near a minimum
- *NOTE*: There may be >1 minimum

Correct Implementation - **Simultaneous** Update $w,b$\
`(EQUATION/ASSIGNMENT)` $tmp_w = w-\alpha\frac{d}{dw}J(w,b)$\
`(EQUATION/ASSIGNMENT)` $tmp_b = b-\alpha\frac{d}{db}J(w,b)$\
`(ASSIGNMENT)` $w = tmp_w$\
`(ASSIGNMENT)` $b = tmp_b$
- Repeat until convergence
- $\alpha$: learning rate (usually a small positive number between 0 and 1)
  - Large $\alpha$: more aggressive descent
  - Small $\alpha$: less agressive descent
- $\frac{d}{dw}J(w,b)$ and $\frac{d}{db}J(w,b)$: derivative of the cost function (gradient vector)







# Unsupervised Learning
`(DEF)` **Unsupervised Learning**: Learning and structuring from data that only comes with inputs (x), but not output labels (y).\
Key characteristic: **Finds something interesting (patterns, structures, clusters, etc.) in unlabeled data -- we don't tell it what's right and wrong**

### Major Types of Unsupervised Learning
`(DEF)` **Clustering**: Groups similar data points together
- Ex: Google News (matching headlines into a recommendation list)

`(DEF)` **Anomaly Detection**: Finds unusual data points

`(DEF)` **Dimensionality Reduction**: Compress data using fewer numbers


