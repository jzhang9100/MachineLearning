Linear Regression

Idea:
   Given a set of features: x1,...,xn try to predict y

Hypothesis Function:
   h0(x) = 0*x0 + x1*01 + ... + xn*0n where x0 = 1
         = Transpose(0) * x - inner product

Cost Function:
   J(0,..0n) = 1/2m sum(h0(xi)-yi)^2 from i=1 to m

Gradient Descent:
   0j = 0j - a * d/d0j J(0) a is step size
   d/d0j = 1/m sum(h0(xi)-y(i)) * xij from i=1 to m where xij is the jth
feature of the ith training example

Feature Scaling:
   Get every feature into approximately -1 <-> 1 range
   Mean normization: xi = (xi - mui)/ std(xi) dont scale x0 since we want
to keep that at 1
   
Example
AutoReply for emails. 
-Input: Contents of an email
-Output: Reply of Yes or No

How we can go about this: 

Preprocessing: Normalization/Tokenization
1. Feature Representation of Contents
 X:
 - 20,000 dimentional vector to represent english corpus.
 - Each index (token) represents the number of occerences of a specific word in
   the input
 - Last token is reserved for words that are out of vocabulary
 Y:
 - Coresponding y values are 1 or 0 for yes or no

Formulation: Logistic Regression Problem
 - Find W such that Wx approximates Y, since Y is either Yes or No
 - [ Probability of Yes|X, Probability of No|X ] with weights w1, w2. We want train our algorithim
   to learn this
 With SGD
 - Iterate through N number of times
 - Sample a random x in X and its corresponding y value
 - If Y == 1 (Yes), update our weights to increase Probability of Yes|X
 - If Y== 0 (No), update our weights to increase Probability of No|X

Implementation of LR on admissions data from Kaggle:
![Loss of implementation](admissions_data_LR_loss.png?raw=true)
