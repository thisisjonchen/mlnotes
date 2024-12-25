# Machine Learning Notes 📝
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/thisisjonchen/mlnotes/main?display_timestamp=author&style=for-the-badge)

My notes from Andrew Ng's "Machine Learning Specialization" 

## Table of Contents
1. [Tools](#tools)
2. [What is Machine Learning?](#what-is-machine-learning)
3. [Supervised Learning](#supervised-learning)
    * 3.1 [Linear Regression](#linear-regression)
      * 3.11 [Cost Function](#cost-function)
      * 3.12 [Gradient Descent](#gradient-descent)
      * 3.13 [Multiple Features](#multiple-features)
      * 3.14 [Vectorization](#vectorization)
      * 3.15 [The Normal Equation](#the-normal-equation)
      * 3.16 [Feature Scaling](#feature-scaling)
      * 3.17 [Checking Gradient Convergence](#checking-gradient-descent-for-convergence)
5. [Unsupervised Learning](#unsupervised-learning)

# Tools
- Language: Python
- Platform: Jupyter Notebook
- Libraries
  - NumPy, scientific computing + lin algebra in Python
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
- The most commonly used cost function for most regression-related models
- `(EQUATION)` $J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( \hat{y}^{(i)} - y^{(i)} \right)^2 = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$
- *NOTE*: Andrew says that the reason we divide by 2 here is to make future calculations "neater"

The squared error cost function is not the only cost function that exists -- there are more.

**The goal** of regression is to minimize the cost function $J(w,b)$
- When we use random $w$, we can get a graph with x-axis $w$ and y-axis $J(w)$ (note, this is excluding $b$ for now to make the example simpler). With this, we can find the minimum $J(w)$ and use it in our model $f$ (2D).
- With both $w, b$, we get a plot where it looks like one part of a hyperbolic paraboloid (or "hammock", "soup bowl", and "curved dinner plate"). This plot would have $b$ and $w$ as parameters/inputs on the bottom axes and $J(w,b)$ on the vertical axis (3D).
  - This can also be accompanied by a contour (topographic) plot with $w$ on the x-axis and $b$ on the y-axis. At the center of the contour plot (where the lines are "growing" from) is where $J(w,b)$ is the minimum.

Now, how can we more easily find the minimum $w,b$? We can use an algorithm called **Gradient Descent**.

### Gradient Descent
One of the most important building blocks in machine learning helps minimize some *any* function.

Outline:
- Start with some $w,b$ (a common approach is to first set $w$=0, $b$=0)
- Keep changing $w,b$ to reduce $J(w,b)$, until we settle at or near a minimum
- *NOTE*: There may be >1 minimum, but with a squared error cost function, it will **never** have multiple local minima (only one global minimum)

Correct Implementation - **Simultaneous** Update $w,b$\
`(EQUATION/ASSIGNMENT)` $tmp_w = w-\alpha\frac{d}{dw}J(w,b)$\
`(EQUATION/ASSIGNMENT)` $tmp_b = b-\alpha\frac{d}{db}J(w,b)$\
`(ASSIGNMENT)` $w = tmp_w$\
`(ASSIGNMENT)` $b = tmp_b$
- Repeat until convergence
- $\alpha$: learning rate (usually a small **positive** number between 0 and 1)
  - *Highly important.* If $\alpha$ is chosen pooorly, the rate of descent may not even work at all.
  - Large $\alpha$: more aggressive descent
    - If too large, it is possible to miss/overshoot the minimum entirely. May fail to converge &#8594; diverge.
  - Small $\alpha$: less agressive descent
    - If too small, the # of updates required will grow significantly, but gradient descent will still work
- $\frac{d}{dw}J(w,b)$ and $\frac{d}{db}J(w,b)$: derivative of the cost function (gradient vector... this is where the "divide by two" from earlier comes in handy)
  - $\frac{d}{dw}J(w,b) = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)})-y^{(i)})x^{(i)}$
  - $\frac{d}{dw}J(w,b) = \frac{1}{m} \sum_{i=1}^{m} (f_{w,b}(x^{(i)})-y^{(i)})$
  - Positive $\frac{d}{dw}J(w,b)$ or $\frac{d}{db}J(w,b)$: $w$ or $b$ decreases slightly
  - Negative $\frac{d}{dw}J(w,b)$ or $\frac{d}{db}J(w,b)$: $w$ or $b$ increases slightly
- **If you are already at a local minimum**, then further gradient descent steps will do nothing.
  - Near a local minimum, the derivative becomes smaller &#8594; update steps become smaller
  - Thus, can reach minimum without modifying/decreasing the learning rate $\alpha$
 
"Batch" Gradient Descent
- "Batch": Each step of gradient descent uses *all the training examples*
- There are other gradient descent algorithms that look at just the subsets


### Multiple Features
Rather than use just one feature $x$, we can increase the number of inputs ("features") that our model considers by using the notation $x_1, x_2, ... x_j$

More Notation:
- $x_j$: $j^{th}$ feature
- $n$: number of features
- $\vec{x}^{(i)}$: number of $i^{th}$ training examples
  - For example, if we had three features, $\vec{x}^{(1)}$ may equal [1416 3 2] (a row matrix)
- $x_{j}^{(i)}$: value of feature $j$ in $i^{(th)}$ training example
  - Using the same values in the previous example, $x_{1}^{(1)}$ = 1416 (1-indexed)

Previously, a univariate linear regression model equation would be $f_{w,b}(x)=wx+b$\
A multiple linear regression equation would be `(EQUATION)` $f_{w,b}(x)=w_1x_1 + w_2x_2 + ... + w_nx_n +b$
- *NOTE*: This is **not the same** as multivariate regression (it is a different thing)
- $\vec{w} = [w_1 \ w_2 \ w_3 ... w_n]$ (parameters of the model)
- $\vec{x} = [w_1 \ w_2 \ w_3 ... w_n]$ (vector)
- $b$: a scalar number
  
Using $\vec{w}$ and $\vec{x}$, we can simplify `(EQUATION)` $f_{w,b}(x)$ = $\vec{w} \cdot \vec{x} + b$ **(Vectorization)**

### Vectorization 
Behind the scenes:
- Without vectorization (e.g. a for loop): will run iteratively
- With vectorization: will run in parallel **MUCH MORE EFFICIENT** (even more efficient with specialized hardware like GPUs with thousands of cores - CUDA)
  - Efficient &#8594; Scale to large datasets

To implement this in Python, we can use **NumPy** with arrays:
```
w = np.array([1.0, 2.5, -3.3])
b = 4
x = np.array([10, 20, 30])
f = np.dot(w,x) + b
```
Will significantly run faster than manually specifying $w[0] * x[0] + ... + w[n] * x[n]$ or with for loop and makes code shorter, especially when $n$ is large.

The cost function can be re-written to include the vector $\vec{w}$: $J(\vec{w}, b)$

This can also be applied to the **gradient descent** algorithm (multiple linear reg. version)
- (repeat)
  - $w_j = w_j - \alpha\frac{d}{dw_j}J(\vec{w},b)$
  - $b = b - \alpha\frac{d}{db}J(\vec{w},b)$ 

In Python:
```
w = np.array([0.5, 1.3, ... 3.4])
d = np.array([0.3, 0.2, ... 0.4])
a = 0.1 # alpha, learning rate
w = w - a * d
```
- $\vec{w} = (w_1 \ w_2 ... w_n)$
- $\vec{d} = (d_1 \ d_2 ... d_n)$ (partial deriv. of the cost function with respect to $b$ and $w$)

### The Normal Equation
Only works for linear regression - solves for $w,b$ *without* iterations

**Disadvantages**
- Does not generalize other learning algorithms
- Slow when the number of features is large (> 10,000)

**What you need to know**
- The normal equation may be used in machine learning libraries that implement linear regression

### Feature Scaling
Rescaling features (assume $\vec{x_1}$ and $\vec{x_2}$) to be more comparable to eachother. This leads the contour plot of $J(\vec{w}, b)$ to be less tall and skinny but rather to be more circular. This allows the path to find the global minimum more directly.\

*In other words*, it is a technique to make gradient descent run much faster.

Ways to feature scale:
- Divide the entire feature range by the maximum:
  - Given the range $300 \le x_1 \le 2000$, $x_{1, scaled} = \frac{x_1}{2000}$
  - Thus, the scaled range would be $0.15 \le x_1 \le 1$
  - The same will be done to $x_2$
- Mean Normalization:
  - Find the average of each feature e.g., ($x_1$) as $\mu_{1}$
  - Given the range $300 \le x_1 \le 2000$, $\mu_{1} = 600$, $x_{1, scaled} = \frac{x_1-\mu_1}{max-min} = \frac{x_1-600}{2000-300}$
  - Thus, the scaled range would be $-0.18 \le x_1 \le 0.82$
  - The same will be done to $x_2$
- Z-Score Normalization:
  - Find standard deviation $\sigma$ and the average of each feature e.g., ($x_1$) as $\mu_{1}$
  - Given the range $300 \le x_1 \le 2000$, $\mu_{1} = 600, \sigma_1 = 450$, $x_{1, scaled} = \frac{x_1-\mu_1}{\sigma_1} = \frac{x_1-600}{650}$
  - Thus, the scaled range would be $-0.67 \le x_1 \le 3.1$
  - The same will be done to $x_2$

Tips for feature scaling:
- Aim for range $-1 \le x_j \le 1$ for each feature $x_j$
  - Scalar multiples of these are fine as well e.g., $-3 \le x_j \le 3$, **as long as they are less than 10 and greater than 0.1**
  - Something like $-1000 \le x_1 \le 1000$ (too large &#8594; rescale)
  - Something like $0.001 \le x_1 \le 0.001$ (too small &#8594; rescale)
- When in doubt, just feature rescale

### Checking Gradient Descent for Convergence






















# Unsupervised Learning
`(DEF)` **Unsupervised Learning**: Learning and structuring from data that only comes with inputs (x), but not output labels (y).\
Key characteristic: **Finds something interesting (patterns, structures, clusters, etc.) in unlabeled data -- we don't tell it what's right and wrong**

### Major Types of Unsupervised Learning
`(DEF)` **Clustering**: Groups similar data points together
- Ex: Google News (matching headlines into a recommendation list)

`(DEF)` **Anomaly Detection**: Finds unusual data points

`(DEF)` **Dimensionality Reduction**: Compress data using fewer numbers


