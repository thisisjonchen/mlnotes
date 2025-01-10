# Machine Learning Notes 📝
![GitHub last commit (branch)](https://img.shields.io/github/last-commit/thisisjonchen/mlnotes/main?display_timestamp=author&style=for-the-badge)

My notes on Andrew Ng's "Machine Learning Specialization" (MLS)

## Table of Contents
1. [Tools](#tools)
2. [What is Machine Learning?](#what-is-machine-learning)
    * 2.1 [Supervised Learning](#supervised-learning)
    * 2.2 [Unsupervised Learning](#unsupervised-learning)
    * 2.3 [Andrew's Thoughts on AGI](#andrews-thoughts-on-agi)
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
      * 3.25 [Overfitting](#overfitting)
      * 3.26 [Regularization](#regularization)
4. [Advanced Learning Algorithms](#advanced-learning-algorithms)
   * 4.1 [Neural Networks](#neural-networks)
      * 4.11 [Recognizing Images](#recognizing-images)
      * 4.12 [Neural Network Model](#neural-network-model)
      * 4.13 [TensorFlow Implementation](#tensorflow-implementation)
      * 4.14 [Training Using TensorFlow](#training-using-tensorflow)
      * 4.15 [Activation Functions](#activation-functions)
      * 4.16 [Multiclass Classification](#multiclass-classification)
      * 4.17 [Multi-Label Classification](#multi-label-classification)
      * 4.18 [Advanced Optimization](#advanced-optimization)
      * 4.19 [Backpropagation](#backpropagation)
   * 4.2 [Advice for Applying ML](#advice-for-applying-ml)
      * 4.21 [Debugging](#debugging)
      * 4.22 [Evaluation](#evaluation)
      * 4.23 [Bias and Variance](#bias-and-variance)
      * 4.24 [Learning Curves](#learning-curves)
      * 4.25 [Iterative ML Development Loop](#iterative-ml-development-loop)
      * 4.26 [Data Engineering](#data-engineering)
      * 4.27 [Full Cycle of an ML Project](#full-cycle-of-an-ml-project)
      * 4.28 [Fairness, Bias, and Ethics](#fairness-bias-and-ethics)
      * 4.29 [Skewed Datasets](#skewed-datasets)
   * 4.3 [Decision Trees](#decision-trees)
      * 4.31 [Decision Tree Model](#decision-tree-model)
      * 4.32 [Learning Process](#learning-process)
      * 4.33 [One-Hot Encoding](#one-hot-encoding)
      * 4.34 [Regression Trees](#regression-trees)
      * 4.35 [Tree Ensembles](#tree-ensembles)
      * 4.36 [XGBoost](#xgboost)
      * 4.37 [When to Use Decision Trees](#when-to-use-decision-trees)
5. [Beyond Supervised Learning](#beyond-supervised-learning)
   * 5.1 [Unsupervised Learning](#unsupervised-learning)
      * 5.11 [Clustering](#clustering)
      * 5.12 [Optimization Objective](#optimization-objective)
      * 5.13 [Anomaly Detection](#anomaly-detection)
      * 5.14 [Anomaly Detection vs. Supervised Learning](#anomaly-detection-vs-supervised-learning)
   * 5.2 [Recommender Systems](#recommender-systems)
      * 5.21 [Making Recommendations](#making-recommendations)
      * 5.22 [Collaborative Filtering](#collaborative-filtering)
      * 5.23 [Binary Labels](#binary-labels)
      * 5.24 [Mean Normalization](#mean-normalization)
      * 5.25 [Content-Based Filtering](#content-based-filtering)
      * 5.26 [Principal Component Analysis](#principal-component-analysis)
   * 5.3 [Reinforcement Learning](#reinforcement-learning)
    
   

# Tools
- Language: Python
- Platform: Jupyter Notebook
- Libraries
  - NumPy, scientific computing + lin algebra in Python
  - Matplotlib, plotting data
  - TensorFlow, machine learning package 
  - keras (integrated into TF 2.0), creates a simple, layer-centric interface to TF
  - SciKit Learn, open source machine learning
    - sklearn.linear_model.SGDRegressor: gradient descent regression model (performs best with normalized features)
    - sklearn.preprocessing.StandardScaler: performs z-score normalization
    - sklearn.linear_model: contains LogisticRegression + LinearRegression
  - XGBoost, decision tree library
    - `from xgboost import XGBClassifier()` for classification
    - `from xgboost import XGBRegressor()` for regression
  - Pandas, data analysis and manipulation

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

***Major Types of Unsupervised Learning***:

`(DEF)` **Clustering**: Groups similar data points together
- Ex: Google News (matching headlines into a recommendation list)

`(DEF)` **Anomaly Detection**: Finds unusual data points

`(DEF)` **Dimensionality Reduction**: Compress data using fewer numbers

## Andrew's Thoughts on AGI
Andrew provided his thoughts on AGI in Part 2 of MLS in "Advanced Learning Algorithms". I wanted to memorialize them here and take inspiration for my growing worldview on AI and, more specifically, artificial general intelligence (AGI).

> "...I think that the path to get there is not clear and could be very difficult. I don't know whether it would take us mere decades and whether we'll see breakthroughs within our lifetimes or it if it may take centuries or even longer to get there." - Andrew Ng


AI includes two very different things: artificial narrow intelligence (ANI) and artificial general intelligence (AGI)

ANI made tremendous progress in recent decades with examples like smart speakers (Siri, Alexa), self-driving cars, etc. - specific applications of intelligence

AGI would be doing anything a human can, but obviously, we are not there yet.

Can we mimic the human brain? No... we have (almost) no idea how the brain works. Every few years, new breakthroughs fundamentally change the way how we perceive the brain. 

But can we see some AGI breakthroughs in our lifetimes? There are glimmers of hope.

**The "one learning algorithm" hypothesis**
Many parts of the brain can adjust depending on what data it is given to function accordingly.

Andrew provides an example of "rewiring" the brain where a part of it (ex, the somatosensory cortex, or the part that handles touch receptors) is instead fed data from a different sensor (ex, eyes with images), the part of the brain (in this case, the somatosensory cortex) learns to function differently (the somatosensory cortex learns to see)

Another is "seeing with your tongue", wherein a grid with varying voltages is applied to a tongue. By mapping a grayscale image to the voltage grid, one can "see" with their tongue even if they are blind

What if we can translate these to a computer? Multimodal developments are already happening... but Andrew stresses to avoid overhyping. No one knows what will happen, but with hard work and time, AGI can happen eventually.


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

### Overfitting
Both linear and logistic regression can work well for many tasks, but sometimes, in an application, the algorithm(s) can run into a problem called overfitting, which can cause it to perform poorly.

What does fitting mean? This refers in context to the best fit line. If we use a linear line on a training set of data points where a quadratic line may have been better, then the linear line does not fit the training set very well (underfit - high bias).

Another extreme scenario would be using some curve that may fit the training set *perfectly* where a simpler curve would suffice (to the point where the line is "wiggly"), and the cost is 0 on all points. If we use this model on an extraneous example, then it could predict the output completely wrong (overfitting - high variance).

`(DEF)` **Underfitting (High Bias)**: Does not fit the training set that well as if the learning algorithm has some sort of strong preconception

`(DEF)` **Generalization (Just Right)**: Fits training set pretty well (best fit)

`(DEF)` **Overfitting (High Variance)**: Fits the training set extremely well, but new examples can result in highly variable predictions

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
- ***NOTE***: Weights + Biases are automatically determined during the training process and will eventually be updated iteratively through a process called backpropagation, which minimizes the loss function using optimization w/ algos like gradient descent (will be discussed more later)

Example: **Demand Prediction**
- `(DEF)` **Activation**: $a = g(z) = \frac{1}{1+e^{-\vec{w} \cdot \vec{x} +b}}$
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
- `(DEF)` **Input Layer ($\vec{x}$)**: Layer 0 containing all the inputs/features we are plugging into the neural network
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

How do you convert pictures to features?
- The picture, if 1000px x 1000px, is actually a 1000 x 1000 matrix of varying pixel intensity values, which range from 0-255
- If we were to "roll" up these values into a singular vector $\vec{x}$, then it would contain *1 million pixels* intensity values
  - How to roll up a matrix? One way is to do L &#8594; R, down one row, then go from R &#8594; L until you are done with the entire matrix
 
A possible process may look like this for facial recognition:\
Input Picture $(\vec{x})$ &#8594; HL1 &#8594; HL2 &#8594; HL3 &#8594; Output Layer &#8594; Probability of being person "XYZ"
- HL 1 finds certain lines (looking at a small window)
- HL 2 groups these lines into certain facial features (looking at a bigger window)
- HL 3 aggregates these facial features into different faces (looking at an even bigger window)
- The Output Layer tries to determine the match probability of identity

A possible process may look like this for car identification:\
Input Picture $(\vec{x})$ &#8594; HL1 &#8594; HL2 &#8594; HL3 &#8594; Output Layer &#8594; Probability of Car Detected
- HL 1 finds certain lines (looking at a small window)
- HL 2 groups these lines into certain car features (looking at a bigger window)
- HL 3 aggregates these car features into different cars (looking at an even bigger window)
- The Output Layer tries to determine the match probability of the car

### Neural Network Model
The fundamental building block of most modern neural networks is a **layer of neurons**
- Every layer inputs a vector of numbers and applies a bunch of logistic regression units to it, and then outputs another vector of numbers (activations) that will be the input into subsequent layers until the final/output layer's prediction of the NN that we then can then threshold
- The *number of layers* includes all hidden layers + output layer, excluding the input layer, and is indexed from 1 (where the input layer is 0)

A neural network layer comprises many neurons, each with its own weight $w$ and bias $b$. These parameters are considered in their respective activations $g(z)$.
- By convention, the activations per layer are denoted by $a^{[i]}$, where $i$ is the index of the particular layer.
- $a^{[1]}$ means the activations from layer 1, $a^{[2]}$ means the activations from layer 2, etc.
- To further differentiate neurons' parameters from different layers, we could use the superscript $[i]$ again, where $w_j^{[i]}$ and $b_j^{[i]}$

The superscript notation continues into the individual scalar activations from each neuron $a$:
```math
\vec{a}^{[1]} = \begin{bmatrix} a_1^{[1]} \\ a_2^{[1]} \\ a_3^{[1]} \end{bmatrix}
```

For a subsequent hidden layers after the first, the $\vec{x}$ becomes the *previous* hidden layer's activations
- Ex: $a_1^{[3]} = g(\vec{w}_1^{[3]} \cdot \vec{a}^{[2]} + b_1^{[3]})$

A general equation for each activation of a neuron (the **Activation Function**) is:\
`(EQUATION)` $a_j^{[l]} = g(\vec{w}_j^{[l]} \cdot \vec{a}^{[l-1]} + b_j^{[l]})$ with the $g(z)$ sigmoid

`(DEF)` **Inference**: Using a trained model to make predictions or classifications on new, unseen data

A popular example of inference is handwritten digit recognition.

`(DEF)` **Forward Propagation**: The process of moving forward from Input Layer &#8594; HL1 &#8594; HL2 &#8594; HL... &#8594; Output Layer from left to right

### TensorFlow Implementation
`(DEF)` **Tensor**: Think of it as a matrix
With TensorFlow + NumPy, we can build these layers:\
Ex: x &#8594; HL1 (3 Neurons) &#8594; Output Layer &#8594; Prediction w/ Threshold
```
x = np.array([200.0, 17.0])
layer_1 = Dense(units=3, activation="sigmoid") # Dense is another name for a layer, units = num. of neurons
a1 = layer_1(x) # apply layer_1 to vector x, where a1 is the activations
### OUTPUT tf.Tensor([[0.2 0.7 0.3]], shape=(1, 3), dtype=float32)
# Translation: 1 x 3 matrix (shape) with type float
# Can use a1.numpy() to convert to numpy matrix

layer_2 = Dense(units=1, activation="sigmoid")
a2 = layer_2(a1) # apply layer_2 to the vector a1 activations from HL1
### OUTPUT tf.Tensor([[0.8]], shape=(1, 1), dtype=float32)
# Translation: 1 x1 matrix (shape) with type float
# The number 0.8 is the activation/prediction from the output layer

# apply threshold (given 0.5)
if a2 >= 0.5:
   yhat = 1
else:
   yhat = 0
```

Note about NumPy Arrays:
- In `np.array([[]])`, every [] is a row, separated by commas. Inside [], each number belongs to a specific column
- *NOTE* It is **double square** brackets (1 set to enclose everything, another for the rows)

For example, `np.array([[1, 2]])` would be a row matrix (**Will be more commonly used for applications like describing features**):
```math
\begin{bmatrix} 1 & 2 \end{bmatrix}
```

Can also be represented as `np.array([[1], [2]])` as a column matrix:
```math
\begin{bmatrix} 1 \\ 2 \end{bmatrix}
```

A 2D matrix with `np.array([1, 2], [4, 5])` looks like this:
```math
\begin{bmatrix} 1 & 2 \\ 4 & 5 \end{bmatrix}
```

Structuring a inference with a Neural Network using TensorFlow:
```
layer_1 = Dense(units=3, activation="sigmoid") # Don't need to explicitly reference 
layer_2 = Dense(units=1, activation="sigmoid") # Don't need to explicitly reference

model = Sequential([Dense(units=3, activation="sigmoid"),
                    Dense(units=1, activation="sigmoid")])
# sequentially strings together these two layers to build a NN
# by convention, we don't use layer declarations, just use Dense() directly

model.compile(...)

x = np.array([200.0, 17.0],
             [120.0, 5.0],
             [425.0, 20.0],
             [212.0, 18.0]) # 4 x 2 matrix
y = np.array([1,0,0,1]) # targets

model.fit(x,y) # tells TF to sequentially string the layers and train it on x, y

model.predict(x_new) # outputs a new prediction based on a new dataset
```

### Training Using TensorFlow
Given a set of $(X,Y)$ examples, how to build and train this in code?\
`(DEF)` **Epochs**: Number of steps in gradient descent

Example Code:
```
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Step 1:  Specifies model telling TF how to compute for the inference
model = Sequential([Dense(units=25, activation="sigmoid"),
                    Dense(units=15, activation="sigmoid"),
                    Dense(units=1, activation="sigmoid")])

# Step 2: Compiles the model using a specific loss function
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss=BinaryCrossentropy()) # example loss function "binary cross entropy"

# Step 3: Trains the model
model.fit(X,Y, epochs=100)
```

**Model Training Step Parallels between Traditional + Neural Networks (TF)**:
1. Define Model: Specify how to compute output given input $x$ and parameters $w,b$
   - Logistic Regression (TRAD)
     - z = np.dot(w,x) + b
     - f_x = 1/(1+np.exp(-z))
   - model = Sequential([Dense(...), \
                    Dense(...),\
                    Dense(...)])  (TF)
2. Specify Loss and Cost Functions
   - Logistic Loss (TRAD)
     - loss = -y * np.log(f_x) - (1-y) * np.log(1-f_x)
   - model.compile(loss=BinaryCrossentropy())  (TF)
3. Train on data to minimize the cost $J(\vec{w}, b)$
   - w = w - alpha * dj_dw  (TRAD)
   - b = b - alpha * dj_db (TRAD)
   - model.fit(X,Y, epochs=100) (TF - .fit() uses **Backpropagation**)
       
`(DEF)` **Binary Cross-entropy**: Also known as logistic loss... binary re-emphasizes either 0 or 1 classification

*NOTE*: Binary cross-entropy is, of course, not the only loss function. For regression-related problems, we could use `MeanSquaredError()`, also imported from tensorflow.keras.losses

### Activation Functions
So far, we have been using the *Sigmoid* function ($g(z) = \frac{1}{1+e^{-z}}$ as our activation function in most of our applications, and it is indeed commonly used particularly in classification applications since $0 < g(z) < 1$.

However, there are more and their usages depend on various applications:
- `(DEF/EQUATION)` **ReLU (Rectified Linear Unit)**: $g(z) = max(0, z)$
   - *NOTE*: $g(z)$ can never be 0 here, but $g(z)$ can be $+\infty$ if $z$ was
- `(DEF/EQUATION)` **Linear Activation Function**: $g(z) = z$
   - As if $g$ was never there at all

To recap, these three are the most commonly used: Sigmoid, ReLU, and Linear AF\
The next question is: how to choose an activation function to use?

**For Binary Classification (y = 0/1)**: Use Sigmoid

**For Regression (y = +/-)**: Use Linear AF

**For Regression (y >= 0)**: Use ReLU 

In reality, **ReLU** is the most common. Why?
- Faster to compute and learn
- $\frac{d}{dw}J(W,B) \approx 0$ when $g(z)$ is flat, which is when $y<0$ for ReLU... speeds up gradient descent
- Use almost *all the time* for hidden layers

**Why do we need activation functions?**
- If we, for example, used all Linear AF for our activation functions (incl. HL + Ouput Layer), this would be no different than linear regression (defeats the purpose of a NN)
- If we used Linear AF on all HL but Sigmoid on Output Layer, then it will be equivalent to logistic regression
- Thus, **do not use linear activations in hidden layers (suggested: ReLU)**


### Multiclass Classification
`(DEF)` **Multiclass Classification**: Refers to classfication problems where you can have more than just two output labels (not just 0/1 like binary)
- The "handwriting digit" example is a good start - classifying not just 0 or 1, but digits from 0-9
- For predictions, the notation still looks like $P(y=n | \vec{x}$ for some $n$, but instead of $n = 0/1$, it can be anything

`(DEF)` **Softmax**: A generalization of logistic regression but with $N$ possible outputs
- `(EQUATION)` $z_j = \vec{w}_j \cdot \vec{x} + b_j$ where $j = 1..., N$
- `(EQUATION)` $a_j = \frac{e^{z_j}}{\sum^{N}_{k=1} e^{z_k}} = P(y= j | \vec{x})$
- *NOTE*: $a_1 + a_2 + ... + a_N = 1$ (100%). Also, with only two outputs, Softmax reduces down to logistic regression but with different parameters
- `(EQUATION)` $loss(a_1,...,a_N, y) = -\log(a_N) if y = N$

The Softmax will replace the Output Layer as a ***Softmax Output Layer*** with the number of neurons (units) being equal to the number of classifications you wish to have

Example Code (not the most optimal version):
```
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Specify the Model
model = Sequential([Dense(units=25, activation="relu"),
                    Dense(units=15, activation="relu"),
                    Dense(units=1, activation="softmax")])

# Step 2: Specify the Loss + Cost
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(loss=SparseCategoricalCrossentropy()) # Note "SparseCategoricalCrossentropy"

# Step 3: Train model
model.fit(X,Y, epochs=100)
```

The reason why the above is *not the best* is because of numerical roundoff errors

A More Numerically Accurate Implementation of Softmax:
```
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Specify the Model
model = Sequential([Dense(units=25, activation="relu"),
                    Dense(units=15, activation="relu"),
                    Dense(units=1, activation="linear")]) # note linear now, NOT softmax

# Step 2: Specify the Loss + Cost
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(loss=SparseCategoricalCrossentropy(from_logits=True)) # notice "from_logits"

# Step 3: Train model
model.fit(X,Y, epochs=100)

# Step 4: Predict
logits = model(X) # no longer outputting a, but rather z (intermediate val)
f_x = tf.nn.softmax(logits)
```

`(DEF)` **Logits**: An intermediate value $z$ where TensorFlow can rearrange terms to make an algorithm more numerically accurate, at the cost of being less readable due to "magic"

We can also use the parameter `from_logits=True` in our logistic regression algorithm to make it more numerically accurate, but it is not totally needed.

### Multi-Label Classification
While seemingly like multiclass classification where we have one output and multiple labels (like instead of 0/1, we can have 0-9), multi-label classification refers to just multiple labels (outputs).

Street Image Example:
- Is there a car? Yes (1)
- Is there a bus? No (0)
- Is there a pedestrian? Yes (1)
- Thus, the target $y$ would be a column matrix/vector
```math
y = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}
```

There are two neural network architectures that may be used here:
1. Train three neural networks, one for each output of car, bus, and pedestrian
2. Train one neural network with three outputs to simultaneously detect car, bus, and pedestrian
   - The output layer here will have three neurons instead of one (using Sigmoid AF)
  
### Advanced Optimization
Recall the learning rate $\alpha$, which is our learning rate. The learning rate before was always up to you, but there are some cons if the $\alpha$ is too large.

There are some algorithms that can dynamically adjust the learning rate (start large then decrease as we approach the minimum) like the **Adam** algorithm

`(DEF)` **Adaptive Moment Estimation (Adam)**: If $w_j$ (or $b$) keeps moving in the same direction, increase $a_j$, else if $w_j$ (or $b$) keeps oscillating, reduce $a$

Example Code:
```
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Specify the Model
model = Sequential([Dense(units=25, activation="relu"),
                    Dense(units=15, activation="relu"),
                    Dense(units=1, activation="linear")])

# Step 2: Specify the Loss + Cost
from tensorflow.keras.losses import SparseCategoricalCrossentropy
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=SparseCategoricalCrossentropy(from_logits=True))

# Step 3: Train model
model.fit(X,Y, epochs=100)
```

Most practitioners use the Adam algorithm rather than the optional gradient descent algorithm.

There are also additional layer types other than the dense layer (where each neuron output is a function of *all* the activation outputs of the previous layer).

Another type of layer often used is ***Convolutional Layers***.

`(DEF)` **Convolutional Layer**: Each neuron only looks at *part* of the previous layer's outputs
- Allows for faster computation
- Need less training data (less prone to overfitting)
- Multiple convolutional layers contribute to a **Convolutional Neural Network** (CNN)
  - Allows for flexibility in window size and opportunity for greater efficiency than NNs with dense layers

### Backpropagation
`(DEF)` **Backpropagation**: A technique that utilizes gradient descent algorithms to more efficiently calculate derivatives as a right-to-left calculation (in relation to a computation graph)
- Also called "autodiff"

The process is that after forward propagation (left-to-right) to compute the cost function, backpropagation (right-to-left) is used for the derivative calculation.

In terms of computational complexity, it would take roughly $N+P$ steps rather than $N \times P$ steps to compute derivates, thus it would make a significant difference in larger neural networks.

Backpropagation is already built into most deep learning algorithms/machine learning frameworks.

## Advice for Applying ML
> "I've seen teams sometimes, say, six months to build a machine learning system, that I think a more skilled team could have taken or done in just a couple of weeks. The efficiency of how quickly you can get a machine learning system to work well will depend to a large part of how well you can repeatedly make good decisions about what to do next in the course of a machine learning project." - Andrew Ng

### Debugging
Say we implemented a regularized linear regression model on housing prices, but it makes unacceptably large prediction errors. What can we try next?
- Get more training examples
- Try smaller sets of features
- Try getting additional features
- Try adding polynomial features($x_1^2, x_1^3, x_1x_2, etc$)
- Try increasing/decreasing the regularization term $\lambda$

Many of these steps can take lots of time, like getting more training examples. How do we know what to do?

`(DEF)` **(Machine Learning) Diagnostic**: A test that you run to gain insight into what is/isn't working with a learning algorithm, to gain guidance into improving its performance
- Can take time to implement, but doing so can be a very good use of your time

### Evaluation
How can one evaluate a model's performance? Evaluating a model's performance properly may lead to debugging, as it may perform worse than expected.

**The 70/30 Technique**: splitting the dataset into two subsets
1. 70% can be used for training the data (($x^{m_{train}}, y^{m_{train}}$) where $m_{train}$ = # of training examples)
2. 30% can be used for testing the data (($x_{test}^{m_{test}}, y_{test}^{m_{test}}$) where $m_{test}$ = # of training examples)

To determine performance, these can be subbed into the cost function $J(\vec{w}, b)$.
- If $J_{train}(\vec{w}, b)$ is low, but $J_{test}(\vec{w}, b)$ is high, then that may indicate that the performance on the training set is good, the performance on general/test set is not as good (overfitting)
- For classification problems,  $J_{train}(\vec{w}, b)$ and $J_{test}(\vec{w}, b)$ indicates the fraction of their respective sets that has been misclassified

However, note that once $\vec{w}, b$ are fit to the training set, the training error $J_{train}(\vec{w}, b)$ is likely lower than the *actual generalization error*. 

In other words, $J_{test}(\vec{w}, b)$ will be a better estimate of how well the model will generalize to new data compared to $J_{train}(\vec{w}, b)$. However, this may also lead to some over-optimistic generalizations on just the test set alone (same problem we had with $J_{train}(\vec{w}, b)$ before). How can we improve upon this?

Further Refinement **60/20/20: Training/Cross-Validation/Test Sets**:
1. 60% can be used for training the data (($x^{m_{train}}, y^{m_{train}}$) where $m_{train}$ = # of training examples)
2. 20% can be used for cross-validating the data (($x_{cv}^{m_{cv}}, y_{cv}^{m_{cv}}$) where $m_{cv}$ = # of training examples)
3. 20% can be used for testing the data (($x_{test}^{m_{test}}, y_{test}^{m_{test}}$) where $m_{test}$ = # of training examples)

`(DEF)` **Cross-Validation**: Extra dataset to check the validity of different models
- Also commonly called the validation set, development set, and dev set

**Model Selection Procedure**:
1. Train the model using the training dataset
2. Test different models with varying degrees $d$ using the cross-validation set and choosing the degree with the lowest cost $J_{cv}(w^{<d>}, b^{<d>})$
3. Estimate the generalization error using the test dataset with the chosen degree: $J_{test}(w^{<d>}, b^{<d>})$

The above was intended for linear regression but also works for choosing a neural network architecture.

**Model Selection Procedure - NN Architecture**:
1. Train the NN using the training dataset
2. Test different $d$ NNs using the cross-validation set and choosing the NN with the lowest cost $J_{cv}(w^{<d>}, b^{<d>})$
3. Estimate the generalization error using the test dataset with the chosen degree: $J_{test}(w^{<d>}, b^{<d>})$

When making decisions regarding your training algorithm, utilize the **training and cv** datasets only. Do not touch the test dataset until you've created one model as your ***final model*** to ensure that the test set is fair and not an overly optimistic estimate of how well a model may generalize new data.


### Bias and Variance
Recall our concepts of **High Bias (Underfit)** and **High Variance (Overfit)**. How might they relate to the costs $J_{train}, J_{cv}$?

($<<$: Much less than)

For high bias (underfit):
- $J_{train}$ is high
- $J_{cv}$ is high
- Relatively, $J_{train} \approx J_{cv}$

For high variance (overfit):
- $J_{train}$ is low
- $J_{cv}$ is high
- Relatively, $J_{train} << J_{cv}$

Both high bias + variance:
- $J_{train}$ is high
- Relatively, $J_{train} << J_{cv}$

If "just right":
- $J_{train}$ is low
- $J_{cv}$ is low

Comparing $J_{train}, J_{cv}$ with the degree of polynomial $d$, while $J_{train}$ will go down as the $d$ increases, $J_{cv}$ has a minimum, but has a point where it will increase again (upward bowl). 

How does bias and variance work with **regularization** ($\lambda$)?
- Very large $\lambda$ = High bias (underfit), where $f_{\vec{w}, b}(\vec{x}) \approx b$ (e.g. $\lambda = 10,000$)
- Very small $\lambda$ = High variance (overfit) (e.g. $\lambda = 0$)

Thus, how can we choose a good $\lambda$? Approach it like the 60/20/20 technique with a polynomial degree, starting from $\lambda = 0, 0.01, ....$ and **doubling the previous** to minimize $J_{cv}$. The notation $(w^{<d>}, b^{<d>})$ depends on just the order $d$ you go (ex: 1. $\lambda = 0.01$) rather than the actual $\lambda$ value as the $d$.

Comparing $J_{train}, J_{cv}$ with the regulation parameter $\lambda$, while $J_{train}$ will go up as $\lambda$ increases, $J_{cv}$ has a minimum, but has a point where it will increase again (upward bowl). 

But now, from what perspective do we assume $J_{train}$ and $J_{cv}$ to be high? First, we need a **baseline level of performance**
- Large gap between baseline performance (error) and training error: high variance (overfit)
- Large gap between training error and cross-validation error: high bias (underfit)

Speech Recognition Example:
- Training error $J_{train}$: 10.8%
- Cross Validation error $J_{cv}$: 14.8%
- Just looking at these two, we may assume high bias from the baseline of 0%, right?
  - However, if we factor in human-level performance (10.6%), we can't assume the learning algorithm to be much better than that
  - Thus, if we consider 10.6% as the baseline, $J_{train}$ is only +0.2% higher. The difference between $J_{train}$ and $J_{cv}$ remains the same (+4%).
  - Now, we may consider this to be a variance problem rather than a bias problem
 
In establishing a baseline level of performance, the question is: "What is the level of error you can *reasonably* hope to get to?"
- Human-level performance
- Competing algorithms performance
- Guess based on experience


### Learning Curves
Learning curves are a way to help you understand how your learning algorithm is doing as a function of the training set $m$ vs. error $J$ (both $J_{train}$ and $J_{cv}$). 

If a learning algorithm has *high bias*:
- The CV error $J_{cv}$ will decrease then plateau
- The training error $J_{train}$ will increase then plateau beneath the CV error
- Both will be relatively higher than the human-level performance, regardless of $m$ **no matter how big $m$ is**
   - If a learning algo suffers from high bias, getting more training data will (by itself) not help much
 
If a learning algorithm has *high variance*:
- The CV error $J_{cv}$ will be much larger than $J_{train}$, decreasing then plateauing as $m$ increases
- The training error $J_{train}$ will increase then plateau
- The "point of plateau" is the human-level performance, which acts like a horizontal asymptote between the two errors
  - However, this means more $m$ = good!
  - If a learning algorithm suffers from high variance, getting more training data is likely to help

Back to our example with housing predictions: Say we implemented a regularized linear regression model on housing prices, but it makes unacceptably large prediction errors. What can we try next? Which will be used if either high variance vs. high bias?
- Get more training examples (fixes high variance)
- Try smaller sets of features (fixes high variance)
- Try getting additional features (fixes high bias)
- Try adding polynomial features($x_1^2, x_1^3, x_1x_2, etc$) (fixes high bias)
- Try increasing/decreasing the regularization term $\lambda$
  - Increase (fixes high variance)
  - Decrease (fixes high bias)
 
If you have high bias, your training set is not the problem - your model is. As such, don't randomly throw away training examples.

**Bias and Variance in Neural Networks**:\
Large neural networks are usually low-bias machines

Simple Process for evaluating bias and variance in neural networks:
- **Does it do well on the training set?** (1)
  - If not ($J_{train}$ is relatively high),  expand the neural network (add more HL or hidden units) until it does well on the training set (achieves comparable to human-level+ performance)
  - **If so, does it do well on the cross-validation set?** (2)
    - If not ($J_{cv}$ is high), add more data and go back to step (1)
    - ***If so, done!***

Understandably, the solutions to (1) and (2) may be hard:
1. GPUs and AI accelerators are speeding this process up, but as NNs get super large, they will get significantly harder to compute
2. Data may be limited due to the nature of the data

A large neural network will usually do as well or better than a smaller one so long as regularization is chosen appropriately (less risk of overfitting than traditional AI). However, they do become more computationally expensive.

To regularize, we add the parameter `kernel_regularizer=(...)` to `Dense(...)` layer.

### Iterative ML Development Loop
*Start*
- Choose architecture (model, data, etc.)
- Train model
- Diagnostics (bias, variance, and error analysis)
*Repeat*

**Example Development Process** - Building an Email Spam Classifier:
1. Choose Architecture
   - Supervised learning
     - $\vec{x}$ = features of email
       - Features: list the top 10,000 words of the email to compute $x_1,x_2,...,x_{10,000}$
     - $y$ = spam (1) or not spam (0)
2. Train model
   - Can use either logistic regression or a neural network to predict $y$ given features $x$
3. Diagnostics
   - How do you try to reduce the error in your spam classifier?
      - Collect more data (e.g., Spam "Honeypot")
      - Develop sophisticated features based on email routing (from the email header)
      - Design algorithms to detect misspellings
4. Repeat if needed

Choosing a ***more promising direction*** can speed up your project many times over than if you chose the wrong path.

`(DEF)` **Error Analysis**: Manually examining misclassified examples and categorizing them based on common traits
- This can help determine what to focus on and what has a high impact on your model
  - In relation to these categories, we may decide to gather more data or features
- These error categories may be overlapping and not mutually exclusive
- For large misclassified examples, just get a small subset (~100+) to examine

### Data Engineering
Some tips/techniques on **engineering the data used by your system**:
- **Adding Data**: Rather than adding more data on everything, add more data on the types where error analysis has indicated it might help
  - E.g., with the email spam classifier, go to *unlabeled* data and find more examples of say, Pharma-related spam if it came up often in error analysis
  - Could boost learning algorithm performance much more than just general data
- `(DEF)` **Data Augmentation**: modifying an *existing* training example to create a new training example
  - Typically used in audio and image classification
    - In image text recognition, it may involve rotations, distortions, etc. (like what you see in captchas)
    - Speech recognition may involve adding noise (e.g., background, bad connection/quality, etc.)
  - However, usually **does not help** to add purely random/meaningless noise to your data
- `(DEF)` **Data Synthesis**: synthetically generate realistic data
  - Often used in computer vision tasks and less for other applications
    - An example is photo-to-text
      - Real data may involve real-world photos
      - Synthetic data may involve typing in a text editor with different fonts

Approaches to developing AI:
- `(DEF)` **Conventional Model-Centric Approach**: AI = ***Code (algorithm/model)*** + Data
   - Significant emphasis on the code
- `(DEF)` **Data-Centric Approach**: AI = Code (algorithm/model) + ***Data***
   - More emphasis on data engineering and collection

`(DEF)` **Transfer Learning**: Using models and data from a different task
- Let's use the example of the handwritten digits classification 0-9 and an image classification NN with 1,000 classes/output units
  - Since the image classification NN has already been trained, we could *re-use* some of its parameters $w, b$ (hidden layers only; output layer is excluded due to size difference) on the digits classification
  - We have two options:
     1. only train output layer parameters
     2. train all parameters

`(DEF)` **Supervised Pretraining**: Training a model on a very large dataset (possibly not a related topic, but a similar application) with the intention of then fine-tuning it to a specific topic
- Downloading a pre-trained model is one way to get a jumpstart on your desired application and topic area
- Use the same input type (images, audio, text, etc.)

To apply transfer learning:
1. Download NN parameters pre-trained on a large data with the same input type (e.g., images, audio, text) as your application (or train your own)
2. Further train (fine-tune) the network on your own data

*Especially* helpful if you lack the resources/availability to get large datasets pertaining to your application if the downloaded NN parameters are relatively similar
- Examples of popular transfer learning applications are GPT-3, BERTs, ImageNet, etc.

### Full Cycle of an ML Project
1. Define Project Scope
2. Define and Collect Data 
3. Train and Perform Error Analysis (*Iterative Improvement with (2) ♻️)
4. Deploy, Monitor, and Maintain System (May need to go back to (2), (3) if needed)

Example Deployment Structure:
- Mobile app (sends API call ($x$) to Inference Server, e.g., audio clip)
- Inference Server (returns Inference ($\hat{y}$) to Mobile App, e.g., text transcript)
  - Contains ML Model
 
Software engineering may be needed (**Machine Learning Operations** (MLOps)):
- Ensure reliable and efficient predictions
- Scaling
- Logging
- System monitoring (new data?)
- Model updates

### Fairness, Bias, and Ethics
Machine learning algorithms today affect ***billions of people***. As such, it is necessary to approach designing machine learning systems with fairness and ethics in mind.

In the past, there were times when ML systems discriminated/biased against various groups:
- Hiring tool that discriminates against women
- Facial recognition system matching dark-skinned individuals to criminal mugshots
- Biased bank loan approvals
- Toxic effect of reinforcing negative stereotypes

There also have been adverse/negative use cases:
- Deepfakes
- Spreading toxic/incendiary speech through optimizing for engagement
- Generating fake content for commercial and political purposes
- Using ML to build products, commit fraud, etc.
  - Spam vs. Anti-Spam; Fraud vs. Anti-Fraud

Possible Guidelines to Maintain Fairness & Ethics
- Get a diverse team to brainstorm things that might go wrong with emphasis on possible harm to vulnerable groups
- Carry out a literature search on standards/guidelines for your industry
- Audit systems against possible harm before deployment
- Develop a mitigation plan (if applicable), and after deployment, monitor for possible harm
  - Simple mitigation: rollback to a previous model where it was reasonably fair
  - Real-life scenario: All self-driving car companies developed mitigation plans for if a car gets into an accident
 
### Skewed Datasets
When working on a machine learning application (particularly classification), the ratio of positive to negative can be very far off from 50:50. In these cases, usual error metrics like accuracy do not work well.

Example - Rare Disease Classification:
- Assume $y=1$ if disease is present, $y=0$ otherwise
- If our training error $J_{train}$ is 1% (99% correct diagnoses), but only 0.5% of patients have the disease, then a simple `print("y=0")` will be more effective since it has an "error" of 0.5% (99.5% correct diagnoses). *A simple print statement outperformed a learning algorithm!*
- However, as stated before, if the dataset is skewed, the accuracy may not be a reliable error metric. Instead, we can use another error metric **pair**: Precision/Recall

To setup for precision/recall, we need to separate the data and predictions into 4 categories:
1. True Positive (Actual = 1, Predicted = 1)
2. False Positive (Actual = 0, Predicted = 1)
3. False Negative (Actual = 1, Predicted = 0)
4. True Negative (Actual = 0, Predicted = 0)

`(DEF)` **Precision**: Proportion of positive predictions that are actually correct; accuracy of positive predictions
- `(EQUATION)` $P = \frac{\textrm{True Positives}}{\textrm{Total Predicted Positive}}$
- Example: Of all patients where we predicted $y=1$, what fraction actually have the rare disease?

`(DEF)` **Recall**: Proportion of actual positive cases that the model correctly identifies; how well the model finds relevant instances in a dataset
- `(EQUATION)` $R = \frac{\textrm{True Positives}}{\textrm{Total Actual Positive}}$
- Example: Of all patients that actually have the rare disease, what fraction did we correctly detect as having it?

A learning algorithm with both low precision and low recall is not a useful algorithm.

However, getting high precision and high recall may not be possible -- there may be a trade-off. This depends on our **classification threshold**.

Suppose still the example of rare diseases and $f(\vec{x})$ is the prediction:
- If the threshold was set at 0.9 (we want to predict $y=1$ if we are very confident)
  - ***Higher precision, lower recall***
- If the threshold was set at 0.3 (we want to avoid missing too many cases of rare diseases when in doubt, predict $y=1$):
   - ***Lower precision, higher recall***

Thresholds are usually up to *you* and depend on the nature of an application.

Luckily, there is one more useful metric that may help us: **The F1 score**

`(DEF)` **F1 Score**: Combines precision ($P$) and recall ($R$) into a single score
- `(EQUATION)` $F_1 = 2 \frac{PR}{P+R}$
- Also called the harmonic mean


## Decision Trees
Decision trees are one of the learning algorithms that are very powerful, widely used in many applications, and even often win in machine learning competitions. Despite this, they lack attention from academia and are not so popular compared to their counterparts, neural networks.

### Decision Tree Model
The "tree" refers to one similar to a binary tree in computer science (not a biological tree in nature). 

The structure is hard to illustrate in words [click here to see illustration](https://miro.medium.com/v2/resize:fit:1400/0*bBSGR61xG55j0fPP.png), but each root has a feature with various options to traverse downward (with binary classification, the example root may be "ear shape" and if "pointy", go left, else if "floppy", go right). The children's roots are **not the specific options**, but a new decision classification (e.g., face shape, whiskers, etc.).

The shapes around each feature have a meaning and may be separated into two categories: decision nodes (ovals) and leaf nodes (rectangles).

`(DEF)` **Decision Nodes**: All other nodes other than the leaf nodes. At each decision node, there are options to cause you to decide either to go left or right down the tree.
- The root node (very top) is also a decision node

`(DEF)` **Leaf Nodes**: The nodes at the very outside edges of the tree. These make a prediction.

### Learning Process
With decision trees, there are some decisions we need to consider on our own:
1. How to choose what feature to split on at each node to maximize purity (or minimize impurity)
2. When do you stop splitting? (Splitting too much runs the risk of overfitting)
   - When a node is 100% one class
   - When splitting a node will result in the tree exceeding a maximum (user-set) depth
   - When improvements in purity score are below a threshold
   - When the number of examples in a node is below a threshold
  
`(DEF)` **Entropy**: A measure of *impurity*
- Commonly with the notation $H(p_1)$, where $p_1$ = fraction of positive examples according to the classification output criteria (e.g., cat or not cat)
- If $p_1$ is the fraction of positive examples, $p_0$ is the fraction of negative examples and is $p_0 = 1 - p_1$
- `(EQUATION)` $H(p_1) = -p_1\log_2(p_1)-(1-p_1)\log_2(1-p_1)$
   - *NOTE*: It is possible to get $0\log(0)$ if $p_1 = 1$ or $p_1 = 0$, and it is known that $\log(0) = - \infty$. However, just assume that this term equals $0$.
- $0 \le H(p_1) \le 1$, where $1$ is the highest entropy (impurity)

`(DEF)` **Information Gain**: Reduction of entropy
- `(EQUATION)` $w^{side} = \frac{\textrm{num of entries on a side}}{\textrm{num of total entries}}$
- `(EQUATION)` Reduction = $H(p_1^{\textrm{root}}) - (w^{\textrm{left}} H(p_1^{\textrm{left}}) + w^{\textrm{right}} H(p_1^{\textrm{right}}))$

**Decision Tree Learning Process**:
1. Start with all examples at the root node
2. Calculate information gain for all possible features and pick the one with **highest information gain**
3. Split the dataset according to the selected feature and create left/right branches of the tree
4. Keep repeating the splitting process until the stopping criteria (1 or more) are met (**recursive splitting**):
   - When a node is 100% one class
   - When splitting a node will result in the tree exceeding a maximum (user-set) depth
   - When improvements in purity score are below a threshold
   - When the number of examples in a node is below a threshold

### One-Hot Encoding
Rather than classifying with just one feature in the decision nodes, we can split that one feature into three or more specific features. This allows for more than two options for a decision.
- E.g., Ear Shape (Pointy/Oval) &#8594; Pointy Ears (0/1), Floppy Ears (0/1), Oval Ears (0/1)
- If a categorical feature *can* take on $k$ values, create $k$ binary features (0/1 valued)

The table becomes a bit more detailed, too, which you can view [here](https://global.discourse-cdn.com/dlai/original/3X/f/0/f0c2c26af0a928152acf23e771de087ac0ecd0bb.jpeg). Note in each row only one column has a hot (1) feature.

One-hot encoding can also be applied to neural networks/logistic regression.

**Continuous Valued Features**:\
In many applications, there are features where there is no binary classification, such as weight. How can we represent them in decision trees? Just as before: splitting. 

The recommended way to find the best split for a feature is to choose $(n-1)$ mid-points between $n$ examples as possible splits and find the split that gives the highest information gain.

### Regression Trees
Rather than taking decision trees as classification algorithms where they make a binary guess of yes (1) or no (0), we can also generalize decision trees to be regression algorithms that can predict a number.

Rather than focusing on reducing entropy, we will instead try to reduce the weighted **variance** of values $Y$ at each subset. 

`(DEF)` **Variance**: How wide a set of numbers is
- `(EQUATION)` $V(\textrm{node}) = \frac{\sum_{i=1}^{n}(x_i - \bar{x})^2}{n-1}$
  - $x_i$ = the value of one observation
  - $\bar{x}$ = the mean value of all observations
  - $n$ = the number of observations
- `(EQUATION)` Weighted Variance Reduction = $V(\textrm{root}) - (w^{\textrm{left}} V(\textrm{left}) + w^{\textrm{right}} V(\textrm{right}))$

### Tree Ensembles
A single decision tree can be highly sensitive to small changes in data. One solution to make the algorithm less sensitive is building many decision trees (a tree ensemble).

`(DEF)` **Tree Ensemble**: A collection of different decision trees
- Q: How can we split features into different decision trees?
  - A: By changing the root feature.
  - *NOTE*: decision nodes can reuse features from different trees in the ensemble
- Each tree makes a binary inference (prediction 0/1) based on classification (e.g., cat or not cat) as a "vote". The majority wins, so an ensemble always has an odd number of trees.

To build a tree ensemble, a technique called **sampling with replacement** is required:
- Imagine putting some colored tokens in a bag, shaking it, and taking one out (also noting what color it was). That is sampling. The "with replacement" part refers to returning the token to the bag for the next iteration.
  - Why is "with replacement" important? Without replacement, the new training set will always be identical to the original training set
- This is needed because it will help construct multiple random training sets that are all slightly different from the original training set 

There are many tree ensemble algorithms, and they work much better than a single decision tree. 

One way of generating a tree sample (Bagged Decision Tree):
- Assume a training set of size $m$
- For $b = 1$ to $B$:
  - Use sampling with replacement to create a new training set of size $m$
  - Train a decision tree on the new dataset
  - Setting $B$ to a large number never hurts performance, but beyond a certain point, you end up with diminishing returns and do not get much larger when $B \approx 100+$. Many recommend 64 to 128 as the maximum.
 
We can improve upon this with the **Random Forest Algorithm**:
- When choosing a feature to split at each node, if $n$ features are available, pick a random subset of $k < n$ features and allow the algorithm to choose only from that subset of features.
- A common choice is to set $k = \sqrt{n}$

### XGBoost
There are many different ways to build decision trees and decision tree ensembles, but by far, the most commonly used implementation of decision trees/ensembles is with an open-source algorithm called **XGBoost**.

`(DEF)` **XGBoost**: Stands for Extreme Gradient Boosting; a scalable gradient-boosted decision tree machine learning library
- Fast, efficient implementation
- Good choice of default splitting criteria and criteria for when to stop splitting
- Built-in regularization to prevent overfitting
- Highly competitive algorithm for ML competitions (e.g., Kaggle)
- `from xgboost import XGBClassifier()` for classification
- `from xgboost import XGBRegressor()` for regression

**Boosted Trees Intuition**:
- Recall our Bagged Decision Tree algorithm. We are going to tweak it slightly under "Use sampling with replacement to create a new training set of size $m$"
- The methodology is similar, but instead of picking from all examples with equal ($\frac{1}{m}$) probability, make it more likely to pick misclassified examples from *previously trained trees*
  - Looks at what we are not doing "quite well at" and tries to build future decision trees to be better at that misclassification problem
 
### When to Use Decision Trees
Decision Trees and Tree Ensembles:
- Works well on tabular (structured/spreadsheet) data
- Not recommended for unstructured data (images, audio, or text)
- Fast to train
- Small decision trees may be human-interpretable
- Most of the time, use XGBoost

Neural Networks:
-  Works well on all types of data, including tabular (structured) and unstructured data
-  May be slower than a decision tree
-  Works with transfer learning
-  When building a system of multiple models working together, it might be easier to string together multiple neural networks


# Beyond Supervised Learning
This section will touch on the following topics:
- Unsupervised Learning
  - Clustering
  - Anomaly Detection
- Recommender Systems
- Reinforcement Learning

## Unsupervised Learning
Recall that unsupervised learning is learning and structuring from data that only comes with inputs (x), not output labels (y). This type of learning is particularly helpful in finding patterns in a data pool (unstructured data) with assistance from an algorithm called **clustering**.

### Clustering
`(DEF)` **Clustering**: An algorithm that looks at a number of data points and automatically finds data points that are related/similar to each other
- Applications of clustering: Grouping similar news, market segmentation (identifying various groups), DNA genetic traits data, etc.

**K-Means Clustering Algorithm**:
- Initialization: Randomly initialize $K$ cluster centroids ($\mu_1, \mu_2, ..., \mu_K$) on an unlabeled training set
   - Choose $K < m$
   - `(DEF)` **Cluster Centroids**: Centers of clusters, initially randomly guessed but will move as more iterations of k-means occur
   - The number of clusters and cluster centroids may be ambiguous
   - Sometimes, the guesses are not the best. We can run the initialization multiple times (maybe ~50 to 1000 times) and compare them using the distortion function (**random initialization**). Select the one with the lowest distortion (cost)
1. Assign each point to its closest **cluster centroid**
2. Recompute centroids by taking an average of its group of points
   - If a cluster had zero points (dividing by 0 points for mean would be undefined), then we just eliminate that cluster
3. Repeat until no points are reassigned (convergence)

K-Means may also be applicable to unlabeled datasets where clusters are not well separated.

### Optimization Objective
In supervised learning, the goal has always been to optimize a cost function with various algorithms like gradient descent. It turns out that clustering is also optimizing a specific cost function, but it is not with gradient descent.

**K-Means Optimization Objective**:
- $c^{(i)}$ = index (from 1 to $K$) of cluster centroid closest to $x^{(i)}$
- $\mu_k$ = cluster centroid $k$
- $\mu_c^{(i)}$ = cluster centroid  of cluster to which example $x^{(i)}$ has been assigned

`(DEF)` **Distortion Function**: Another name for cost function in clustering
- `(EQUATION)` $J(c^{(1)}, ..., c^{(m)}, \mu_1, ..., \mu_K) = \frac{1}{m} \sum_{i=1}^{m} || x^{(i)} - \mu_{c^{(i)}} || ^2$
- As this function is being optimized, it is *guaranteed* to go down or stay the same on each step of k-means. If it goes up, there is likely a bug in the code.

**Choosing a value of $K$**:
- `(DEF)` **Elbow Method**: Take the cost function $J$ as a function of $K$ clusters and find where the point decrease of $J$ where it starts plateauing. That point of $K$ should be your answer.
   - Andrew does not endorse this method
   - Don't choose $K$ just to minimize cost $J$, since the $J$ always decreases
- Often, you want to get clusters for some later (downstream) purpose. Evaluate K-means based on how well it performs on that later purpose
   - E.g., T-Shirt sizes, select $K=3$ for sizes S, M, and LG
 
### Anomaly Detection
Whereas clustering algorithms group similar events/values, anomaly detection looks at an unlabeled dataset of normal events and thereby learns to detect if there is an unusual or anomalous event.

How can we develop such an algorithm? With a technique called **density estimation**.

**Density Estimation**:
- Helps determine the probability of $x$ ( $p(x)$ )being seen in the dataset by determining regions of high probability (denser regions of $x$) and low probability (more sparse regions of $x$)
- $\epsilon$ = probability threshold
  - As we decrease $\epsilon$, the algorithm is *less likely* to detect an anomaly
- $p(x_{test}) < \epsilon$ = potential anomaly
- $p(x_{test}) \ge \epsilon$ = ok (normal)

Example Applications of Anomaly Detection:
- Fraud detection:
  - $x^{(i)}$ = features of user $i$'s activities (how often logged in, how many pages visited, transactions, etc.)
  - Model $p(x)$ from data
  - Identify unusual users by checking which have $p(x) < \epsilon$
- Manufacturing
  - $x^{(i)}$ = features of product $i$ (airplane engines, circuits, phones)
    
**Gaussian Distribution**\
Also known as the normal distribution, it will be useful in density estimation
- Assume $x$ is a number
- Probability of $x$ is determined by a Gaussian with mean $\mu$ (signifying the central point in the curve) with variance $\sigma^2$
- Contains a bell-shaped curve, $\sigma$ is the standard deviation
- Area under the curve always sums up to 1
- `(EQUATION)` $p(x) = \frac{1}{\sqrt{2\pi} \sigma} e^{\frac{-(x-\mu)^2}{2\sigma^2}}$
- `(EQUATION)` $\mu = \frac{1}{m} \sum_{i=1}^m x^{(i)}$
- `(EQUATION)` $\sigma^2 = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu)^2$

**Anomaly Detection Algorithm**:
- Assume training set { $\vec{x}^{(1)}, \vec{x}^{(2)} ,... \vec{x}^{(m)}$ } where each example $x^{(i)}$ has $n$ features
- `(EQUATION)` $p(\vec{x}) = p(x_1; \mu_1, \sigma_1^2) * p(x_2; \mu_2, \sigma_2^2) * ... p(x_n; \mu_n, \sigma_n^2) = \Pi_{j=1}^n p(x_j ; \mu_j, \sigma_j^2)$
  - Assumes the features $x_1, x_2, ..., x_m$ are statistically independent but work fine even if they are dependent
1. Choose $n$ features $x_i$ that you think might be indicative of anomalous examples
2. Fit parameters $\mu_1, ... \mu_n, \sigma_1^2, ... \sigma_n^2$ (use equations from gaussian distribution)
3. Given new example $x$, compute $p(x)$
4. Flag anomaly if $p(x) < \epsilon$

**The Importance of Real-Number Evaluation**
- When developing a learning algorithm (choosing features, etc.), making decisions is much easier if we have a way of evaluating our learning algorithm (also called real-number evaluation)
- Assume we have some labeled data of anomalous and non-anomalous examples (y=1/0)
- We will be using the supervised learning idea of labeled data with CV and test sets to evaluate our algorithm (only a small portion of anomalies)
  - The training set will have only normal (non-anomalous) examples and will remain unlabeled
  - The CV and test sets will include a few anomalous examples but mostly normal examples
- Use CV set to choose parameter $\epsilon$

**Choosing Features**:
- Replace highly non-Gaussian features with Gaussian features as our Gaussian probability functions $p(x)$ are more likely to fit and find anomalies with Gaussian features
  - Some potential functions to try to replace $x$: $log(x), log(x+c), x^{\frac{1}{2}}, x^{\frac{1}{n}}$
  - $c$ = some random constant (add a small number like 0.0001 if min(x) is 0 for log)
  - Whatever transformation you apply to the training set, apply also to the CV and test sets as well
- Q: With anomaly detection, there is a threshold $\epsilon$ for $p(x)$ to separate anomalous and non-anomalous examples, but what if both are large and comparable to one another?
  - A: Add new features

### Anomaly Detection vs. Supervised Learning
Above, we discussed using *some* labeled data in the evaluation of our anomaly detection algorithm. Why not just go for supervised learning?

When to use Anomaly Detection:
- Very small number of positive examples/anomalies ( $y=1$ ) (0-20 is common)
- Large number of negative ( $y=0$ ) examples
- Many different "types of anomalies; it is hard for any algorithm to learn from positive examples what the anomalies look like
- Future anomalies may look nothing like any of the previous anomalous examples (unpredictable in the future)
- Ex: fraud detection, manufacturing (find new previously unseen defects), monitoring machines in data center

When to use Supervised Learning:
- Large number of positive and negative examples
- Enough positive examples for the algorithm to get a sense of what positive examples are like
- Future positive examples likely to be similar to ones in the training set (predictable in the future)
- Ex: Email spam classification, manufacturing (find previously seen defects), weather prediction, disease classification


## Recommender Systems
Recommender systems are often used in industry applications like Amazon and Netflix but received far less attention in academia. Their impact is undeniable; they are directly responsible for a large fraction of sales in finance and e-commerce today.

### Making Recommendations
Let's use this [table](https://github.com/user-attachments/assets/790b2e92-4500-484a-8c4c-d45164c858dc) of movie ratings to illustrate recommender systems
- $r(i,j)$ = 1 if user $j$ has rating movie $i$ (0 otherwise)
- $n_u$ = no. of users
- $n_m$ = no. of movies
- $y^{(i,j)}$ = rating given by user $j$ on movie $i$ (if defined)
- $w^{(j)}, b^{(j)}$ = parameters for user $j$
- $x^{(i)}$ = features for movie $i$
  - Individual features may be hard to interpret
  - To find other items related to it, find item $k$ with $x^{(k)}$ similar to $x^{(i)}$
    - $\sum_{i=1}^n (x_l^{(k)} - x_l^{(i)})^2$
- For user $j$ and movie $i$, predict rating using linear regression (as there is a simple relationship): $f_{w,b}(x) = w^{(j)} \cdot x^{(i)} + b^{(j)}$
- $m^{(j)}$ = no. of movies rated by user $j$
- Goal: to learn $w^{(j)}, b^{(j)}$
- Cost Function with regularization parameter for user $j$, $w^{(j)}, b^{(j)}$, recall: $\frac{1}{2m^{(j)}} \sum_{i:r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2m^{(j)}} \sum_{k=1}^n  (w_k^{(j)})^2$
  - Notice the limiter $i:r(i,j)=1$ under the summation, which limits to user-rated movies only

Cost Function with regularization parameter for **all users and parameters** $w^{(1)}, b^{(1)}, w^{(2)}, b^{(2)}, ..., w^{(n_u)}, b^{(n_u)}$: 

`(EQUATION)` $J(w^{(1)},...,w^{(n)}, b^{(1)}, ..., b^{(n)}) = \frac{1}{2} \sum_{j=1}^{n_u} \sum_{i:r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n (w_k^{(j)})^2$

What if we didn't have the features that describe the items of the movies in sufficient detail (like romance and action)? We can use an algorithm called **Collaborative Filtering**

### Collaborative Filtering
Using the same example from before, we can predict features $x^{(i)}$ since we have parameters from the *same* users and *same* movies. This allows us to infer feature vectors from data. Typical linear regression can't come up with features from scratch as it relies on predefined, explicit input features.

In short, we can learn the input features!

Given $w^{(1)}, b^{(1)}, w^{(2)}, b^{(2)}, ..., w^{(n_u)}, b^{(n_u)}$, cost function to learn $x^{(i)}$:

`(EQUATION)` $J(x^{(i)} = \frac{1}{2} \sum_{j:r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{k=2}^{n} (x_k^{(i)})^2$

Cost function to learn $x^{(1)}, x^{(2)}, ... , x^{(n_m)}$:

`(EQUATION)` $J(x^{(1)},...,x^{(n)}) = \frac{1}{2} \sum_{i=1}^{n_m} \sum_{j:r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n (x_k^{(i)})^2$

If we put the cost function for all parameters and the cost function for all features together:

`(EQUATION)` $J(x^{(1)},...,x^{(n)}, w^{(1)},...,w^{(n)}, b^{(1)}, ..., b^{(n)}) = \frac{1}{2} \sum_{(i,j):r(i,j)=1} (w^{(j)} \cdot x^{(i)} + b^{(j)} - y^{(i,j)})^2 + \frac{\lambda}{2} \sum_{j=1}^{n_u} \sum_{k=1}^n (w_k^{(j)})^2 + \frac{\lambda}{2} \sum_{i=1}^{n_m} \sum_{k=1}^n (x_k^{(i)})^2$

Then, we can perform a gradient descent, where the cost function is now $J(w,b,x)$. Also, we want to minimize three parameters now: $w, b, x$, where we take the partial derivatives, respectively.

**Limitations of Collaborative Filtering**:
- "Cold Start" problem: How to rank new items that few users have rated? Or show something reasonable to new users who have rated few items?
- Not able to use side information about users or items, only one (like rating)
  - Item ex: Genre, movie stars, studio, ...
  - User: Demographics (age, gender, location), expressed preferences, ...

### Binary Labels
Many important applications of recommender systems involve binary labels, not just a rating from 0-5.
- Example with item recommendation:
  - Did the user $j$ purchase an item? (0/1/?)
   Did the user $j$ like an item? (0/1/?)
  - Did the user $j$ spend at least 30 secs. with an item? (0/1/?)
  - Did the user $j$ click on an item? (0/1/?)
- Meaning of ratings
  - 1: engaged after being shown
  - 0: did not engage after being shown
  - item not yet shown
 
Now, instead of regression, we use binary classification (logistic regression).

Previously: predict $y^{(i,j)}$ as $w^{(j)} \cdot x^{(i)} + b^{(j)}$

For binary labels, we use the logistic function: predict that the probability of $y^{(i,j)} = 1$ is given by $g(w^{(j)} \cdot x^{(i)} + b^{(j)})$ where $g(z) = \frac{1}{1+e^{-z}}$

### Mean Normalization
From back in supervised learning, we saw that normalizing can help the algorithm run faster and more efficiently.

Using the movie recommendation example from earlier, the normalization process computes the average rating of each movie row and then groups them in a vector $\mu$.
- Instead of the ratings being 0-5, we *subtract* the vector $\mu$ from each row and their respective elements (some elements, such as originally 0, can be negative... this will be handled in the prediction)
- For user $j$ on movie $i$, predict $w^{(j)} \cdot x^{(i)} + b^{(j)} + \mu_i$

### Content-Based Filtering
*Collaborative* filtering may recommend items to you based on ratings of users who gave similar ratings as you. *Content-based* filtering may recommend items to you based on the features of the user and the item to find a good match.
- Will continue to use $r^{(i,j)} = 1$ to indicate if $j$ has rated item $i$ and $y^{(i,j)}$ as rating given by user $j$, but with more features
- We denote the feature vectors by $x_u^{(j)}$ for user $j$ and $x_m^{(i)}$ for movie $i$, and these vectors could vary in size in comparison to each other

Predict the rating of user $j$ on movie $i$ as: 
- `(EQUATION)` $v_u^{(j)} \cdot v_m^{(i)}$ 
- $v_u^{(j)}$ computed from $x_u^{(j)}$
- $v_m^{(i)}$  computed from $x_m^{(i)}$
- Why "computed"? Since we are taking the dot product, both $v$ vectors have to be the same size, whereas their $x$ counterparts could have been different
- This is like the linear regression algorithm but without the $b$

How do we compute the $v$ vectors from $x$? One way is to use a **deep learning algorithm** (neural networks)
- The number of units in each HL does not matter, but the output layer must have the same size/dimension
- Then, we process it through a cost function:
  - `(EQUATION)` $J = \sum_{(i,j):r(i,j)} (v_u^{(j)} \cdot v_m^{(i)} - y^{(i,j)}) ^2$ + NN regularization term
  - Trains all the parameters of the user and movie networks

How can you efficiently generate a recommendation from a large set of items? Two steps: **Retrieval & Ranking**
- `(DEF)` **Retrieval**: Generate a large list of plausible item candidates, then combine retrieved items into list, removing duplicates and items already watched/purchased
  - e.g., For each of the last 10 movies watched by user, find the 10 most similar movies
  - Retrieving more items results in better performance but slower recommendations
  - To analyze/optimize the trade-off, carry out *offline* experiments to see if retrieving additional items results in more relevant recommendations (i.e., $p(y^{(i,j)} = 1$ of items displayed to the user are higher)
- `(DEF)` **Ranking**: Takes the list retrieved and ranks using the learned model (NN), then display ranked items to the user


### Principal Component Analysis
`(DEF)` **Principal Component Analysis (PCA)**: An unsupervised learning algorithm that is commonly used for visualization, specifically in applications with lots of features (hard to plot, say, 50 features)
- The goal is to reduce the number of features to 2-3 to graph and visualize
- To get there, PCA finds a new axis and coordinates from, many features (e.g., car length and height &#8594; size). The axis is not in another dimension, but "overlaid" the graph and coordinates of the original features, confusingly called the "z-axis"
- Notation also changes, like on axes, we go from original features $x_1, x_2, ..., x_n$ to maybe $z_1, z_2$ after PCA

When we choose an axis, we **project** the coordinates onto this new axis ("z-axis")
- How do we choose an axis? We may draw a "line" to where the variance is *large*(points spread out) to where we are capturing important info from the original data
- If the *variance* is small (points squished together), then that is an indicator of a bad axis selection
- When we achieve the max variance possible, then the axis is called the **principal component**

We can have 2-3 of these principal component axes. How so? By drawing the additional axes at a perpendicular (90 deg) angle to the 1st principal component axis.

*NOTE*: PCA is not linear regression. Linear regression attempts to minimize distance in the vertical direction (ground truth y-axis). In contrast, PCA treats the axes equally and tries to minimize distance according to a "z-axis" to maximize variance.

Now, let's go the other way. How can we find the (approximate) original values of $x$ given $z$? With a technique called **reconstruction**.

`(DEF)` **Reconstruction**: Approximates original values of $(x_1,x_2)$ using the "z-axis" vector.
- We multiply $z$ by the vector containing only vector lengths of $x_1, x_2$

**PCA Algorithm**:
1. Optional Pre-processing: perform feature scaling
2. "Fit" the data to obtain 2 (or 3) new axes (principal components) (we can use sklearn's `fit` to do this automatically. Includes mean normalization)
3. Optionally examine how much variance is explained by each principal component (using sklearn's `explained_variance_ratio`) 
4. Transform (project) the data onto new axes

## Reinforcement Learning
Reinforcement Learning (RL) is not widely applied in commercial applications today but is one of the pillars of machine learning. RL is not classified under supervised learning or unsupervised learning but is its own category.
- Examples of applications used today: controlling robots, factory optimization, financial (stock) trading, playing games (incl. video games)

The task is to find a **reward** function that maps a state $s$ &#8594; action $a$. What makes RL so powerful is that you have to tell it ***what to do*** rather than how to do it.

Example Scenario with an Autonomous Helicopter:
- Positive reward: Helicopter flying well (+1)
- Negative reward: Helicopter flying poorly (-1000)

Why not use supervised learning? For example, when controlling a robot, it is very difficult to obtain a data set of $x$ and the *ideal* action $y$. 



