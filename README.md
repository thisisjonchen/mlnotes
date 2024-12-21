# Machine Learning Notes 📝
My notes from "Machine Learning Specialization" by Andrew Ng

## Table of Contents
1. [What is Machine Learning?](#what-is-machine-learning)
2. [Supervised Learning](#supervised-learning)
3. [Unsupervised Learning](#unsupervised-learning)

## What is Machine Learning?
Arthur Samuel, a pioneer in CS + AI in his time, defined machine learning as "...[the] field of study that gives computers the ability to learn without being explicitly programmed" (1959). He helped evolve AI by writing the first checkers-playing game that learned from thousands of games against itself.

Machine Learning Algorithms
- Supervised Learning (*Used in most real-world applications/rapid advancements)
- Unsupervised Learning
- Recommender Systems
- Reinforcement Learning


## Supervised Learning
`(DEF)` Supervised Learning: Learning via input (x) to output (y) mappings.\
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
  
`(DEF)` **Classification**: Predict categories from a small number of possible outputs
- Ex: Breast Cancer Detection (Benign vs. Not-Benign)
- Terminology: Classes/Categories are often used interchangeably with the output
- Graphing technique utilizes a boundary line depending on input(s), separating one class/category from another


## Unsupervised Learning
`(DEF)` Unsupervised Learning: Learning and structuring from data that only comes with inputs (x), but not output labels (y).\
Key characteristic: **Finds something interesting (patterns, structures, clusters, etc.) in unlabeled data -- we don't tell it what's right and wrong**

### Major Types of Unsupervised Learning
`(DEF)` Clustering: Groups similar data points together
- Ex: Google News (matching headlines into a recommendation list)

`(DEF)` Anomaly Detection: Finds unusual data points

`(DEF)` Dimensionality Reduction: Compress data using fewer numbers


