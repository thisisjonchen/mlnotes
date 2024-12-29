# Machine Learning Notes 📝
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/thisisjonchen/mlnotes/main?display_timestamp=author&style=for-the-badge)

My notes from Andrew Ng's "Machine Learning Specialization" 

## Table of Contents
1. [Tools](#tools)
2. [What is Machine Learning?](#what-is-machine-learning)
    * 2.1 [Supervised Learning](#supervised-learning)
    * 2.2 [Unsupervised Learning](#unsupervised-learning)
3. [Supervised Learning: Regression and Classification](#supervised-learning-regression-and-classification)
    * 3.1 [Linear Regression](#linear-regression)
      * 3.11 [Cost Function](#cost-function)
      * 3.12 [Gradient Descent](#gradient-descent)
      * 3.13 [Multiple Features](#multiple-features)
      * 3.14 [Vectorization](#vectorization)
      * 3.15 [The Normal Equation](#the-normal-equation)
      * 3.16 [Feature Scaling](#feature-scaling)
      * 3.17 [Checking Gradient Convergence](#checking-gradient-descent-for-convergence)
      * 3.18 [Choosing Learning Rate](#choosing-the-learning-rate)
      * 3.19 [Feature Engineering](#feature-engineering)
   * 3.2 [Logistic Regression](#logistic-regression)
      * 3.21 [The Sigmoid Function](#the-sigmoid-function)
      * 3.22 [Decision Boundary](#decision-boundary)
      * 3.23 [Cost Function for Logistic Regression](#cost-function-for-logistic-regression)
      * 3.24 [Gradient Descent for Logistic Regression](#gradient-descent-for-logistic-regression)
   * 3.3 [Overfitting](#overfitting)
      * 3.31 [Addressing Overfitting](#addressing-overfitting)
      * 3.32 [Regularization](#regularization)
4. [Advanced Learning Algorithms](#advanced-learning-algorithms)
   * 4.1 [Neural Networks](#neural-networks)
      * 4.11 [Recognizing Images](#recognizing-images)
   

# Tools
- Language: Python
- Platform: Jupyter Notebook
- Libraries
  - NumPy, scientific computing + lin algebra in Python
  - Matplotlib, plotting data
  - SciKit Learn, open source machine learning
    - sklearn.linear_model.SGDRegressor: gradient descent regression model (performs best with normalized features)
    - sklearn.preprocessing.StandardScaler: performs z-score normalization
    - sklearn.linear_model: contains LogisticRegression + LinearRegression

# What is Machine Learning?
Arthur Samuel, a pioneer in CS + AI in his time, defined machine learning as "...[the] field of study that gives computers the ability to learn without being explicitly programmed" (1959). He helped evolve AI by writing the first checkers-playing game that learned from thousands of games against itself.

Machine Learning Algorithms:
- Supervised Learning (*Used in most real-world applications/rapid advancements)
- Unsupervised Learning
- Recommender Systems
- Reinforcement Learning

## Supervised Learning
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

***Major Types of Supervised Learning***
`(DEF)` **Regression**: Predict a number from infinitely many possible outputs
- Ex: Housing Market Prices (Sq. Feet to Market Value)
- Graphing technique utilizes a best-fitting line (linear, logarithmic, etc.)
- Models: Linear Regression
  
`(DEF)` **Classification**: Predict categories from a small number of possible outputs
- Ex: Breast Cancer Detection (Benign vs. Not-Benign)
- Terminology: Classes/Categories are often used interchangeably with the output
- Graphing technique utilizes a boundary line depending on input(s), separating one class/category from another

## Unsupervised Learning
`(DEF)` **Unsupervised Learning**: Learning and structuring from data that only comes with inputs (x), but not output labels (y).\
Key characteristic: **Finds something interesting (patterns, structures, clusters, etc.) in unlabeled data -- we don't tell it what's right and wrong**

***Major Types of Unsupervised Learning***
`(DEF)` **Clustering**: Groups similar data points together
- Ex: Google News (matching headlines into a recommendation list)

`(DEF)` **Anomaly Detection**: Finds unusual data points

`(DEF)` **Dimensionality Reduction**: Compress data using fewer numbers

# Supervised Learning: Regression and Classification
## Linear Regression
`(DEF)` **Linear Regression**: Fits a best-fitting, straight (linear) line to your data

`(DEF)` **Univariate Linear Regression**: Fancy name for linear regression with one variable (single  x)

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
Rescaling features (assume $\vec{x_1}$ and $\vec{x_2}$) to be more comparable to eachother. This leads the contour plot of $J(\vec{w}, b)$ to be less tall and skinny but rather to be more circular. This allows the path to find the global minimum more directly.

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
Objective: $min_{w,b}J(\vec{w},b)$\
We can check the learning curve by plotting # of iterations (x-axis) with $J(\vec{w},b)$
- $J(\vec{w},b)$ should **decrease** after every iteration
- If the cost function does not decrease, then the alpha chosen was not viable
- The # of iterations will vary from application to application

Automatic Convergence Test:
- Let $\epsilon$ be $10^{-3}$
- If $J(\vec{w},b)$ decreases by $\le \epsilon$ in *one iteration*, declare **convergence**
- However, finding the right $\epsilon$ can be hard. Therefore, using the graph can be a more reliable method for determining convergence (look if $J(\vec{w},b)$ flattens out)

### Choosing the Learning Rate
If the graph of # of iterations (x-axis) with $J(\vec{w},b)$ does not decrease for every iteration, then there are two possible errors:
- Bug in code
- Learning rate $\alpha$ is **too large** (too small will just make # of iterations larger)

How to tell if it is a bug or a learning rate problem: set learning rate $\alpha$ to a very small number. If $J$ still does not decrease on every iteration, then it is a sign that there is a bug.

Values of $\alpha$ to try (Try to find the largest learning rate where $J$ still decreases on every iteration):\
$... 0.001 \quad 0.01 \quad 0.1 \quad 1 ...$

### Feature Engineering
`(DEF)` **Feature Engineering**: Using intuition to design new features by transforming or combining original features.

Why?
The choice of features can have a **huge impact** on a learning algorithm's performance. Thus, choosing the right features is critical to making an algorithm work well.

Example: Predicting the price of a house
- Assume we have and equation $f(x) = w_1x_1 + w_2x_2 + b$, where $x_1$ is the frontage width and $x_2$ is the depth.
- While this is fine, we can refine this by taking the $x_3$ area = frontage * depth. The resulting equation could look like $f(x) = w_1x_1 + w_2x_2 + w_3x_3 + b$
- Note that we did *not* remove features but added one to the equation

**Polynomial Regression**\
Using the ideas of multiple linear regression + feature engineering, we can develop a new algorithm called polynomial regression.

**Important**: As the features are being raised to some power, *feature scaling* becomes increasingly more important. We can take the feature $x$ to any power, including but not limited to $\sqrt{x}$ or $x^{3}$

## Logistic Regression
**Motivation**: Since linear regression is not so good for *classification*-related problems, a logistic regression algorithm is widely used today. While "logistic regression" contains the word "regression", it is used more for classification.

Example of a classification task: Decide if an animal is a cat or not a cat\
Example of a regression task: Estimate the weight of a cat based on its height

Why is linear regression bad for classification? This is because any single outlier may skew the linear regression model to where it may falsely identify a category (misclassification).

*NOTE*: Classes and categories are often used interchangeably here.

`(DEF)` **Binary Classification**: Output $y$ can only be one of two values (e.g., true or false; yes or no; 0 or 1...)

`(DEF)` **Negative Class**: Those that connotate "false" or "absence" != bad

`(DEF)` **Positive Class**: Those that connotate "true" or "presence" != good

`(DEF)` **Decision Boundary**: A "line" that separates classes/categories.

### The Sigmoid Function
Also referred to as a "logistic function", it looks like an "S" on a 2D graph. Outputs between 0 and 1: 

`(EQUATION)` $g(z)=\frac{1}{1+e^{-z}}$ where $0 < g(z) < 1$
- Notice if z = big positive number, the fraction becomes 1. If z = big negative number, the fraction becomes 0.

If we set $z = \vec{w} \cdot \vec{x} + b$, then we can get $g(\vec{w} \cdot \vec{x} + b)$

This becomes our logistic regression model:
`(EQUATION)` $g(\vec{w} \cdot \vec{x} + b) = \frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}}$

In essence, it helps **determine the "probability" that class is 1**

Example: 
- $x$ is "tumor size"
- $y$ is 0 (not malignant) or 1 (malignant)
- $f(\vec{x})$ = 0.7 means that there is a 70% chance that $y$ is 1

Notation used in research example: $f_{\vec{w}, b}(\vec{x}) = P(y=1 | \vec{x};\vec{w},b)$
- Translation: Probability that $y$ is 1, given input $\vec{x}$, parameters $\vec{w}, b$

### Decision Boundary
Now, given the probability, how do we decide at which probability would classify the input as 0 or 1?

A common choice is to choose 0.5 as the threshold: Is $f_{\vec{w}, b} \ge 0.5$?
- Yes: $\hat{y} = 1$
- No: $\hat{y} = 0$
- *NOTE*: With the thresholds, they do not always have to be 0.5, but it is usually better to choose a lower threshold (circumstances matter) and risk a false positive than a misclassification.

To find the decision boundary, we take $z = \vec{w} \cdot \vec{x} + b = 0$ and solve to get the $x$'s on one side and a constant on the other. For more than one feature $x$, this would create a line separating $y=1$ and $y=0$ "clusters".

This can also be applied to *non-linear* decision boundaries where we apply polynomial regression to logistic regression.

Non-linear Decision Boundary Example:
- Set $z = w_1x_1^2 + w_2x_2^2 + b$, where $w_1,w_2=1, b=-1$
- Thus $f(\vec{x}) = g(w_1x_1^2 + w_2x_2^2 + b)$, where $z=x_1^2+x_2^2=1$
- $x_1^2+x_2^2=1$ is conveniently a circle where inside the circle, $y = 0$, and outside, $y = 1$

### Cost Function for Logistic Regression
Compared to the *squared error* cost function we have been using for linear regression, in which its graph would result in a **convex** (bowl) shape, if we apply that to logistic regression, the graph would be **non-convex** (wiggly), meaning that there would be "valleys" that would disrupt our gradient descent algorithm.

A cost function for logistic regression can be defined as such:
- Recall the squared error cost function: $J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$
- `(EQUATION)` **Logistic Cost Function**: $J(w, b) = \frac{1}{m} \sum_{i=1}^{m} [L(f_{\vec{w},b}(\vec{x}^{(i)}, y^{(i)}))]$
- `(DEF)` **Logistic Loss Function**: $L(f_{\vec{w},b}(\vec{x}^{(i)}, y^{(i)}))$
  - Equals $-\log(f_{\vec{w},b}(\vec{x}^{(i)}))$ if $y^{(i)} = 1$
  - Equals $-\log(f_{\vec{w},b}(1 - \vec{x}^{(i)}))$ if $y^{(i)} = 0$
  - Loss is lowest when $f_{\vec{w},b}(\vec{x}^{(i)}, y^{(i)})$ predicts close to true label $y^{(i)}$
    - If $y^{(i)} = 1$...
      - As $f_{\vec{w},b}(\vec{x}^{(i)}, y^{(i)})$ &#8594; 1, then loss &#8594; 0 (Good!)
      - As $f_{\vec{w},b}(\vec{x}^{(i)}, y^{(i)})$ &#8594; 0, then loss &#8594; $\infty$ (bad)
    - Conversely, if $y^{(i)} = 0$...
      -  As $f_{\vec{w},b}(\vec{x}^{(i)}, y^{(i)})$ &#8594; 0, then loss &#8594; 0 (Good!)
      -  As $f_{\vec{w},b}(\vec{x}^{(i)}, y^{(i)})$ &#8594; 1, then loss &#8594; $\infty$ (bad)

`(EQUATION)` **Simplified Loss Function**: $L(f_{\vec{w},b}(\vec{x}^{(i)}, y^{(i)})) = -y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)})) -(1-y^{(i)})\log(f_{\vec{w},b}(1 - \vec{x}^{(i)}))$
- Is completely equivalent to our previous loss function

`(EQUATION)` **Logistic Cost Function Using the Simplified Loss Function**: $J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} [-y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)})) -(1-y^{(i)})\log(f_{\vec{w},b}(1 - \vec{x}^{(i)}))]$

Of course, this is *not* the only cost function. This cost function is derived from **maximum likelihood estimation**.

### Gradient Descent for Logistic Regression
Just like the gradient decent for linear regression, the process remains the same:\
`(EQUATION/ASSIGNMENT)` $tmp_w = w-\alpha\frac{d}{dw}J(w,b)$\
`(EQUATION/ASSIGNMENT)` $tmp_b = b-\alpha\frac{d}{db}J(w,b)$\
`(ASSIGNMENT)` $w = tmp_w$\
`(ASSIGNMENT)` $b = tmp_b$
- Repeat until convergence (simultaneous updates)

The partial derivatives also remain the same:
- $\frac{d}{d\vec{w}}J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(x^{(i)})-y^{(i)})x^{(i)}$
- $\frac{d}{db}J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(x^{(i)})-y^{(i)})$

However, the *only* change remains in the definition of $f(\vec{x})$, where $f_{\vec{w}, b}(\vec{x}) = \frac{1}{1+e^{-(\vec{w} \cdot \vec{x} + b)}}$

Other techniques that can also be applied:
- Learning curve adjustments
- Feature scaling
- Vectorization

## Overfitting
Both linear and logistic regression can work well for many tasks, but sometimes, in an application, the algorithm(s) can run into a problem called overfitting, which can cause it to perform poorly.

What does fitting mean? This refers in context to the best fit line. If we use a linear line on a training set of data points where a quadratic line may have been better, then the linear line does not fit the training set very well (underfit - high bias).

Another extreme scenario would be using some curve that may fit the training set *perfectly* where a simpler curve would suffice (to the point where the line is "wiggly"), and the cost is 0 on all points. If we use this model on an extraneous example, then it could predict the output completely wrong (overfitting - high variance).

`(DEF)` **Underfitting (High Bias)**: Does not fit the training set that well as if the learning algorithm has some sort of strong preconception

`(DEF)` **Generalization (Just Right)**: Fits training set pretty well (best fit)

`(DEF)` **Overfitting (High Variance)**: Fits the training set extremely well, but new examples can result in highly variable predictions

### Addressing Overfitting
How can we address overfitting?

Some options:
- Collect more training examples (the training algo will fit a curve that is "less wiggly" over time)
- Use fewer features
  - If too many features + insufficient training data, it may result in overfit
  - As such, select the most relevant features (**feature selection**)
    - A disadvantage with feature selection is that useful features could be lost, if all features are indeed useful
- `(DEF)` **Regularization**: Reduces the size of parameters $w_j$
  - Gently reduces the impact of some features without removing them completely

### Regularization
As stated in its definition, we reduce the size of parameters $w_j$ to move toward a simpler model that is less likely to overfit.

If we don't know what term to penalize, we can penalize *all* of them a bit. 

Let's use start by **regularizing linear regression's cost function** &#8594; $J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2 + \frac{\lambda}{2m} \sum_{j=1}^n w_j^2$
- The symbol $\lambda$ is called the regularization parameter, where $\lambda > 0$
- $\frac{\lambda}{2m} \sum_{j=1}^n w_j^2$ is called the **regularization term**, which keeps $w_j$ small
- Increasing $\lambda$ will tend to decrease the size of parameters $w_j$, while decreasing it will increase the size of parameters $w_j$
- We also divide $\lambda$ by $2m$ to make it easier to choose a good value for $\lambda$, as it is scaled the same as the cost function as the training set grows
- The inclusion of parameter $b$ with the $\lambda$ makes little difference, thus it is excluded

Regularizing Gradient Descent for Linear Regression:
- The process of gradient descent will remain the same, but the partial derivatives will change to include the new regularization term
- $\frac{d}{d\vec{w}}J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(x^{(i)})-y^{(i)})x^{(i)} + \frac{\lambda}{m}w_j$
- $\frac{d}{db}J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(x^{(i)})-y^{(i)})$ (remains the same since $b$ does not have a significant effect)

Regularizing Logistic Regression:
- Like the regularized cost function for linear regression, we just have to add a regularization term to **logistic regression's cost function** &#8594; $J(w, b) = -\frac{1}{m} \sum_{i=1}^{m} [-y^{(i)}\log(f_{\vec{w},b}(\vec{x}^{(i)})) -(1-y^{(i)})\log(f_{\vec{w},b}(1 - \vec{x}^{(i)}))] + \frac{\lambda}{2m} \sum_{j=1}^n w_j^2$
- The intent is still the same: prevents the size of some parameters $w_j$ from becoming too large
- Just like before, the gradient descent algorithm remains the same:
  - $\frac{d}{d\vec{w}}J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(x^{(i)})-y^{(i)})x^{(i)} + \frac{\lambda}{m}w_j$
  - $\frac{d}{db}J(\vec{w},b) = \frac{1}{m} \sum_{i=1}^{m} (f_{\vec{w},b}(x^{(i)})-y^{(i)})$ (remains the same since $b$ does not have a significant effect)
  - However, recall that the definition of $f$ changes



# Advanced Learning Algorithms
This section will touch on:
- Neural Networks (Deep Learning)
  - Inference (prediction)
  - Training
- Practical advice for building ML systems
- Decision Trees

## Neural Networks
Origins: Ambition to develop algorithms that try to mimic the brain, but in modern NNs, we are shifting away from the idea of mimicking biological neurons
- Also known as **deep learning**
- Used in various applications today, from speech to images (CV) to text (NLP) and more

Why neural networks?
- As the amount of data increased in a wide range of applications, traditional AI like linear regression and logistic regression failed to scale up in performance, increasingly.
- A small neural network trained on that same dataset as traditional AI would see some performance gains -- even more so as we scale up to medium to large-sized neural networks
  - The "size" of a neural network depends on the number of artificial "neurons" it has

Example: **Demand Prediction**
- `(DEF)` **Activation**: $a = f(x) = \frac{1}{1+e^{-wx+b}}$
  - Imagine this like a single neuron in the brain; this accepts an input $x$ and outputs $a$
  - Notice that the function is the same as logistic regression
  - Example: Take $x$ as the price, $a$ being the probability of being a top seller (0 or 1)

Now, let's expand the number of features to price, shipping cost, marketing, and material to determine the probability of being a top seller.

We can "combine" some of these features into one neuron or several neurons in a **layer**:
- `(DEF)` **Layer**: Grouping of neurons that take as input the same or similar features and that, in turn, outputs a few numbers *together*
  - Can have multiple or a singular neuron
  - We can create the first layer as such:
    - Price, Shipping Cost &#8594; Neuron (Affordability)
    - Marketing &#8594; Neuron (Awareness)
    - Price, Material &#8594; Neuron (Perceived Quality)
  - Then, these feed into a second layer, which contains only a singular neuron:
    - Affordability, Awareness, Perceived Quality &#8594; Neuron (Probability of Being a Top Seller)
- `(DEF)` **Input Layer ($\vec{x}$)**: The "pre"-first layer containing all the inputs/features we are plugging into the neural network
  - In the example, "Price", "Shipping Cost", "Marketing", and "Material" would all be in the input layer
- `(DEF)` **Hidden Layer ($\vec{a}$)**: Intermediary layers between input + output
  - In the example, this would include the layer containing "Affordability" and "Awareness"
  - There could be **more than 1** hidden layer, where the number of neurons could vary to be larger than features or smaller than the features
- `(DEF)` **Output Layer ($a$)**: The final layer that outputs our prediction
  - In the example, the second layer would be the output layer
- `(DEF)` **"Activations"**: Refers to the output from a neuron
  - In the example, "Affordability" and "Awareness" would be activations from the first layer
  - "Probability of Being a Top Seller" would be the activation from the final neuron
 
In reality, all neurons in one layer would be able to access **all** features from the previous layer
- For example, the neuron "Affordability" would take all inputs of "Price", "Shipping Cost", "Marketing", and "Material"
- We also do NOT define features or neurons in the hidden layer -- the neural network determines what it wants to use in the hidden layer(s), which is what makes it so powerful

Process:\
Input Layer ($\vec{x}$) &#8594; Hidden Layer(s) ($\vec{a}$) &#8594; Output Layer ($a$) &#8594; Probability of Being a Top Seller

`(DEF)` **Multilayer Perceptron**: Refers to neural networks with multiple hidden layers, often used in literature

**The question with architecting a Neural Network**: How many hidden layers and units do we need?


### Recognizing Images
One application of a neural network is in **computer vision** (CV), which takes as an input a picture and outputs what you may want to know, like maybe the identity of a profile picture.

How to convert pictures to features?
- The picture, if 1000px x 1000px, is actually a 1000 x 1000 matrix of varying pixel intensity values which range from 0-255
- If we were to "roll" up these values into a singular vector $\vec{x}$, then it would contain *1 million pixels* intensity values
  - How to roll up a matrix? One way is to do L &#8594; R, down one row, then go from R &#8594; L until you are done with the entire matrix
 
A possible process may look like so for facial recognition:\
Input Picture $(\vec{x})$ &#8594; HL 1 &#8594; HL 2 &#8594; HL3 &#8594; Output Layer &#8594; Probability of being person "XYZ"
- HL 1 finds certain lines (looking small window)
- HL 2 groups these lines into certain facial features (looking at a bigger window)
- HL 3 aggregates these facial features into different faces (looking at even bigger window)
- The Output Layer tries to determine the match probability of identity

A possible process may look like so for car identification:\
Input Picture $(\vec{x})$ &#8594; HL 1 &#8594; HL 2 &#8594; HL3 &#8594; Output Layer &#8594; Probability of Car Detected
- HL 1 finds certain lines (looking small window)
- HL 2 groups these lines into certain car features (looking at a bigger window)
- HL 3 aggregates these car features into different cars (looking at even bigger window)
- The Output Layer tries to determine the match probability of identity
