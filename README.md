# DecisionTree1


#  Decision Tree Classifier (From Scratch)

A robust, pure Python implementation of a Decision Tree Classifier. 

This project was built **from first principles** (without using pre-built models like `sklearn.tree`) to deeply understand the mathematical engines of Machine Learning: **Entropy**, **Information Gain**, and **Recursive Splitting**.

![Project Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen)

##  The "Why"
Modern ML libraries are "black boxes." To truly understand how Artificial Intelligence makes decisions, I reverse-engineered the logic:
* **No shortcuts:** Every node split, threshold check, and prediction is hand-coded.
* **Math-heavy:** Implements Shannon's Entropy and Information Gain directly.
* **Recursive Logic:** Mimics how human reasoning breaks complex problems into smaller "Yes/No" questions.

## How It Works
The algorithm recursively partitions the data by finding the "Best Split" at every step.

### 1. The Metric: Entropy 
The model measures the "impurity" or chaos of a dataset using the formula:
$$H(S) = - \sum p(x) \log_2 p(x)$$

### 2. The Decision: Information Gain 
The tree iterates through **every feature** and **every threshold** to maximize the gain:
$$IG = H(\text{Parent}) - \text{Weighted Avg} \cdot H(\text{Children})$$

### 3. The Architecture
* **`Node` Class:** Acts as the fundamental unit, storing the feature index, threshold, and pointers to left/right children.
* **`DecisionTree` Class:** Manages the recursion depth (`max_depth`) and fitting process.

##  Installation & Usage

### Dependencies
You only need `numpy` for the math calculations and `pandas` for data handling.
```bash
pip install numpy pandas scikit-learn
