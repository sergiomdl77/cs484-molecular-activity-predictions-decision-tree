# cs484-molecular-activity-predictions-decision-tree
CS 484 Homework 2
cs484-drug-activity-predictions-dec-tree
Overview and Assignment Goals: The objectives of this assignment are the following: Use/implement a feature selection/reduction 
technique. Experiment with various classification models. Think about dealing with imbalanced data. F1 Scoring Metric
Detailed Description:
Develop predictive models that can determine given a particular compound whether it is active (1) or not (0). Drugs are typically
small organic molecules that achieve their desired activity by binding to a target site on a receptor. The first step in the 
discovery of a new drug is usually to identify and isolate the receptor to which it should bind, followed by testing many small 
molecules for their ability to bind to the target site. This leaves researchers with the task of determining what separates the 
active (binding) compounds from the inactive (non-binding) ones. Such a determination can then be used in the design of new 
compounds that not only bind, but also have all the other properties required for a drug (solubility, oral absorption, lack of 
side effects, appropriate duration of action, toxicity, etc.).
The goal of this competition is to allow you to develop predictive models that can determine given a particular compound whether 
it is active (1) or not (0). As such, the goal would be to develop the best binary classification model.
A molecule can be represented by several thousands of binary features which represent their topological shapes and other 
characteristics important for binding.
