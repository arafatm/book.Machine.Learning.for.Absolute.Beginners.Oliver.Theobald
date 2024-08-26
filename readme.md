---
author:Oliver Theobald
title: Machine Learning for Absolute Beginners
---

Machine Learning For Absolute Beginners: A Plain English Introduction
LOCATION: 3

2       WHAT IS MACHINE LEARNING?
LOCATION: 112

key characteristic of machine learning is the concept of self-learning. This refers to the application of statistical modeling to detect patterns and improve performance based on data and empirical information; all without direct programming commands.
LOCATION: 130

By decoding complex patterns in the input data, the model uses machine learning to find connections without human help.
LOCATION: 155

Another distinct feature of machine learning is the ability to improve predictions based on experience.
LOCATION: 157

spam detection system is lured into producing a false-positive based on previous input data.
LOCATION: 172

Traditional programming is highly susceptible to this problem because the model is rigidly defined according to pre-set rules.
LOCATION: 173

Machine learning, on the other hand, emphasizes exposure to data as a way to refine the model, adjust weak assumptions, and respond appropriately to unique data points
LOCATION: 173

Training & Test Data
LOCATION: 181

the training data, which is the initial reserve of data used to develop the model.
LOCATION: 184

After you have developed a model based on patterns extracted from the training data and you are satisfied with the accuracy of its predictions, you can test the model on the remaining data, known as the test data.
LOCATION: 189

The Anatomy of Machine Learning
LOCATION: 194

Machine learning, data mining, artificial intelligence, and computer programming all fall under the umbrella of computer science,
LOCATION: 199

Figure 4: Visual representation of the relationship between data-related fields
LOCATION: 214

data mining focuses on analyzing input variables to predict a new output, machine learning extends to analyzing both input and output variables.
LOCATION: 237

Table 1: Comparison of techniques based on the utility of input and output data/variables
LOCATION: 246

3       MACHINE LEARNING CATEGORIES
LOCATION: 264

Supervised Learning
LOCATION: 272

Supervised learning imitates our own ability to extract patterns from known examples and use that extracted insight to engineer a repeatable outcome.
LOCATION: 273

This process of understanding a known input-output combination
LOCATION: 277

Input data is referred to as the independent variable (uppercase “X”),
LOCATION: 279

the output data is called the dependent variable (lowercase “y”).
LOCATION: 279

Table 2: Extract of a used car dataset
LOCATION: 287

With access to the selling price of other similar cars, the supervised learning model can work backward to determine the relationship between a car’s value (output) and its characteristics (input). The input features of your own car can then be inputted into the model to generate a price prediction.
LOCATION: 288

Figure 5: Inputs (X) are fed to the model to generate a new prediction (y)
LOCATION: 291

When building a supervised learning model, each item (i.e. car, product, customer) must have labeled input and output values—known in data science as a “labeled dataset.”
LOCATION: 294

Figure 6: Labeled data vs. unlabeled data
LOCATION: 297

common algorithms used in supervised learning include regression analysis (i.e. linear regression, logistic regression, non-linear regression), decision trees, k-nearest neighbors, neural networks, and support vector machines,
LOCATION: 298

Unsupervised Learning
LOCATION: 302

In the case of unsupervised learning, the output variables are unlabeled, and combinations of input and output variables aren’t known.
LOCATION: 302

Unsupervised learning instead focuses on analyzing relationships between input variables and uncovering hidden patterns that can be extracted to create new labels regarding possible outputs.
LOCATION: 303

The advantage of unsupervised learning is that it enables you to discover patterns in the data that you were unaware of—such
LOCATION: 312

Unsupervised learning is especially compelling in the domain of fraud detection—where the most dangerous attacks are those yet to be classified.
LOCATION: 313

One interesting example is DataVisor; a company that has built its business model on unsupervised learning. Founded in 2013 in California, DataVisor protects customers from fraudulent online activities, including spam, fake reviews, fake app installs, and fraudulent transactions.
LOCATION: 314

traditional solutions analyze chains of activity for a specific type of attack and then create rules to predict and detect repeat attacks. In this case, the dependent variable (output) is the event of an attack, and the independent variables (input) are the common predictor variables of an attack.
LOCATION: 321

a model that monitors combinations of independent variables, such as a large purchasing order from the other side of the globe or a landslide number of book reviews that reuse existing user content generally leads to a better prediction.
LOCATION: 337

In supervised learning, the model deconstructs and classifies what these common variables are and designs a detection system to identify and prevent repeat offenses.
LOCATION: 339

Sophisticated cybercriminals, though, learn to evade these simple classification-based rule engines by modifying their tactics.
LOCATION: 340

leverage unsupervised learning techniques to address these limitations.
LOCATION: 345

The drawback, though, of using unsupervised learning is that because the dataset is unlabeled, there aren’t any known output observations to check and validate the model, and predictions are therefore more subjective than those coming from supervised learning.
LOCATION: 353

Semi-supervised Learning
LOCATION: 358

used for datasets that contain a mix of labeled and unlabeled cases.
LOCATION: 359

One technique is to build the initial model using the labeled cases (supervised learning) and then use the same model to label the remaining cases (that are unlabeled) in the dataset.
LOCATION: 361

Reinforcement Learning
LOCATION: 366

the third and most advanced category of machine learning.
LOCATION: 367

The goal of reinforcement learning is to achieve a specific goal (output) by randomly trialing a vast number of possible input combinations and grading their performance.
LOCATION: 369

Q-learning
LOCATION: 377

A specific algorithmic example of reinforcement learning
LOCATION: 378

you start with a set environment of states, represented as “S.”
LOCATION: 378

In the game Pac-Man, states could be the challenges, obstacles, or pathways
LOCATION: 379

The set of possible actions to respond to these states is referred to as “A.” In Pac-Man, actions are limited to left, right, up, and down movements, as well as multiple combinations thereof.
LOCATION: 381

The third important symbol is “Q,” which is the model’s starting value and has an initial value of “0.”
LOCATION: 382

As Pac-Man explores the space inside the game, two main things happen: 1) Q drops as negative things occur after a given state/action. 2) Q increases as positive things occur after a given state/action.
LOCATION: 383

more comprehensive explanation of reinforcement learning and Q-learning using the Pac-Man case study. https://inst.eecs.berkeley.edu/~cs188/sp12/projects/reinforcement/reinforcement.html
LOCATION: 390

4       THE MACHINE LEARNING TOOLBOX
LOCATION: 394

Compartment 1: Data
LOCATION: 406

As a beginner, it’s best to start with (analyzing) structured data. This means that the data is defined, organized, and labeled in a table, as shown in Table 3.
LOCATION: 408

Images, videos, email messages, and audio recordings are examples of unstructured data as they don’t fit into the organized structure of rows and columns.
LOCATION: 409

Contained in each column is a feature. A feature is also known as a variable, a dimension or an attribute—
LOCATION: 415

Rows are sometimes referred to as a case or value,
LOCATION: 419

Figure 7: Example of a tabular dataset
LOCATION: 422

Each column is known also as a vector.
LOCATION: 423

Figure 8: The y value is often but not always expressed in the far-right vector
LOCATION: 428

Scatterplots, including 2-D, 3-D, and 4-D plots, are also packed into the first compartment of the toolbox with the data.
LOCATION: 429

Compartment 2: Infrastructure
LOCATION: 437

which consists of platforms and tools for processing data.
LOCATION: 438

Jupyter Notebook)
LOCATION: 439

of machine learning libraries, including NumPy, Pandas, and Scikit-learn,
LOCATION: 439

server. In addition, you may need specialized libraries for data visualization such as Seaborn and Matplotlib, or a standalone software program like Tableau,
LOCATION: 443

Compartment 3: Algorithms
LOCATION: 481

You can find hundreds of interesting datasets in CSV format from kaggle.com.
LOCATION: 483

Beginners typically start out using simple supervised learning algorithms such as linear regression, logistic regression, decision trees, and k-nearest neighbors. Beginners are also likely to apply unsupervised learning in the form of k-means clustering and descending dimension algorithms.
LOCATION: 487

Visualization
LOCATION: 491

The visual story conveyed through graphs, scatterplots, heatmaps, box plots, and the representation of numbers as shapes make for quick and easy storytelling.
LOCATION: 493

The Advanced Toolbox
LOCATION: 499

Beginners work with small datasets that are easy to handle and downloaded directly to one’s desktop as a simple CSV file.
LOCATION: 502

Advanced users, though, will be eager to tackle massive datasets, well in the vicinity of big data.
LOCATION: 503

Compartment 1: Big Data
LOCATION: 506

Compartment 2: Infrastructure
LOCATION: 513

in 2009, Andrew Ng and a team at Stanford University made a discovery to link inexpensive GPU clusters to run neural networks consisting of hundreds of millions of connected nodes.
LOCATION: 525

TensorFlow is only compatible with the Nvidia GPU card,
LOCATION: 531

Compartment 3: Advanced Algorithms
LOCATION: 537

While Scikit-learn offers a range of popular shallow algorithms, TensorFlow is the machine learning library of choice for deep learning/neural networks.
LOCATION: 543

Written in Python, Keras is an open-source deep learning library that runs on top of TensorFlow, Theano, and other frameworks, which allows users to perform fast experimentation in fewer lines of code.
LOCATION: 547

It is, however, less flexible in comparison to TensorFlow and other libraries.
LOCATION: 549

Developers, therefore, will sometimes utilize Keras to validate their decision model before switching to TensorFlow to build a more customized model.
LOCATION: 549

5       DATA SCRUBBING
LOCATION: 560

datasets need upfront cleaning and human manipulation before they’re ready for consumption.
LOCATION: 564

Feature Selection
LOCATION: 570

it’s essential to identify which variables are most relevant to your hypothesis or objective.
LOCATION: 571

Table 4: Endangered languages, database: https://www.kaggle.com/the-guardian/extinct-languages
LOCATION: 575

Let’s say our goal is to identify variables that contribute to a language becoming endangered. Based on the purpose of our analysis, it’s unlikely that a language’s “Name in Spanish” will lead to any relevant insight. We can therefore delete this vector (column) from the dataset.
LOCATION: 577

Secondly, the dataset contains duplicated information in the form of separate vectors for “Countries” and “Country Code.”
LOCATION: 581

Another method to reduce the number of features is to roll multiple features into one,
LOCATION: 583

Table 5: Sample product inventory
LOCATION: 584

For instance, we can remove individual product names and replace the eight product items with fewer categories or subtypes.
LOCATION: 590

Table 6: Synthesized product inventory
LOCATION: 594

The downside to this transformation is that we have less information about the relationships between specific products.
LOCATION: 597

Row Compression
LOCATION: 605

In addition to feature selection, you may need to reduce the number of rows
LOCATION: 605

Table 7: Example of row merge
LOCATION: 609

non-numeric and categorical row values can be problematic to merge while preserving the true value of the original data. Also, row compression is usually less attainable than feature compression and especially for datasets with a high number of features.
LOCATION: 621

One-hot Encoding
LOCATION: 623

you next want to look for text-based values that can be converted into numbers.
LOCATION: 624

with non-numeric data.
LOCATION: 626

one-hot encoding, which transforms values into binary form,
LOCATION: 627

Table 8: Endangered languages
LOCATION: 632

Table 9: Example of one-hot encoding
LOCATION: 641

Using one-hot encoding, the dataset has expanded to five columns, and we have created three new features from the original feature
LOCATION: 643

Binning
LOCATION: 656

(also called bucketing)
LOCATION: 656

used for converting continuous numeric values into multiple binary features called bins or buckets according to their range of values.
LOCATION: 657

Let’s take house price evaluation as an example. The exact measurements of a tennis court might not matter much when evaluating house property prices;
LOCATION: 661

numeric measurements of the tennis court with a True/False feature or a categorical value such as “small,” “medium,” and “large.”
LOCATION: 665

Another alternative would be to apply one-hot encoding with “0” for homes that do not have a tennis court
LOCATION: 666

Normalization
LOCATION: 669

normalization and standardization help to improve model accuracy when used with the right algorithm.
LOCATION: 670

The former (normalization) rescales the range of values for a given feature into a set range with a prescribed minimum and maximum, such as [0, 1] or [−1, 1].
LOCATION: 670

this technique helps to normalize the variance among the dataset’s features which may otherwise be exaggerated by another factor.
LOCATION: 672

Normalization, however, usually isn’t recommended for rescaling features with an extreme range as the normalized range is too narrow to emphasize extremely high or low feature values.
LOCATION: 675

Standardization
LOCATION: 677

This technique converts unit variance to a standard normal distribution with a mean of zero and a standard deviation (σ) of one.[15] This
LOCATION: 678

Figure 11: Examples of rescaled data using normalization and standardization
LOCATION: 681

Standardization is generally more effective than normalization when the variability of the feature reflects a bell-curve shape of normal distribution and is often used in unsupervised learning.
LOCATION: 683

Standardization is generally recommended when preparing data for support vector machines (SVM), principal component analysis (PCA), and k-nearest neighbors (k-NN).
LOCATION: 685

Missing Data
LOCATION: 688

Missing values in your dataset can be equally frustrating and interfere with your analysis and the model’s predictions.
LOCATION: 690

There are, however, strategies to minimize the negative impact of missing data.
LOCATION: 690

One approach is to approximate missing values using the mode value.
LOCATION: 691

This works best with categorical and binary variable types, such as one to five-star rating systems and positive/negative drug tests respectively.
LOCATION: 693

Figure 12: A visual example of the mode and median respectively
LOCATION: 696

The second approach is to approximate missing values using the median value,
LOCATION: 697

This works best with continuous variables, which have an infinite number of possible values, such as house prices.
LOCATION: 698

As a last resort, rows with missing values can be removed altogether. The obvious downside to this approach is having less data to analyze and potentially less comprehensive insight.
LOCATION: 700

6       SETTING UP YOUR DATA
LOCATION: 703

After cleaning your dataset, the next job is to split the data into two segments for training and testing, also known as split validation.
LOCATION: 707

usually 70/30 or 80/20.
LOCATION: 709

training data should account for 70 percent to 80 percent
LOCATION: 710

20 percent to 30 percent of rows are left for your test data.
LOCATION: 711

Before you split your data, it’s essential that you randomize the row order. This helps to avoid bias in your model,
LOCATION: 717

After randomizing the data, you can begin to design your model and apply it to the training data.
LOCATION: 722

The next step is to measure how well the model performed.
LOCATION: 726

Area under the curve (AUC) – Receiver Operating Characteristic (ROC)[16], confusion matrix, recall, and accuracy are four examples of performance metrics used with classification tasks
LOCATION: 727

mean absolute error and root mean square error (RMSE) are commonly used to assess models that provide a numeric output
LOCATION: 730

Using Scikit-learn, mean absolute error is found by inputting the X values from the training data into the model and generating a prediction for each row in the dataset.
LOCATION: 733

You’ll know that the model is accurate when the error rate for the training and test dataset is low, which means the model has learned the dataset’s underlying trends and patterns.
LOCATION: 735

Cross Validation
LOCATION: 744

While split validation can be effective for developing models using existing data, question marks naturally arise over whether the model can remain accurate when used on new data.
LOCATION: 744

Rather than split the data into two segments (one for training and one for testing), you can implement what’s called cross validation.
LOCATION: 748

Cross validation maximizes the availability of training data by splitting data into various combinations and testing each specific combination.
LOCATION: 749

The first method is exhaustive cross validation, which involves finding and testing all possible combinations
LOCATION: 751

The alternative and more common method is non-exhaustive cross validation, known as k-fold validation.
LOCATION: 753

Figure 14: k-fold validation
LOCATION: 761

This method, though, is slower because the training process is multiplied by the number of validation sets.
LOCATION: 768

How Much Data Do I Need?
LOCATION: 770

machine learning works best when your training dataset includes a full range of feature combinations.
LOCATION: 772

At an absolute minimum, a basic machine learning model should contain ten times as many data points as the total number of features.
LOCATION: 784

there is a natural diminishing rate of return after an adequate volume of training data (that’s widely representative of the problem) has been reached.
LOCATION: 789

For datasets with less than 10,000 samples, clustering and dimensionality reduction algorithms can be highly effective, whereas regression analysis and classification algorithms are more suitable for datasets with less than 100,000 samples.
LOCATION: 791

Neural networks require even more samples to run effectively and are more cost-effective and time-efficient for working with massive quantities of data.
LOCATION: 793

Scikit-learn has a cheat sheet for matching algorithms to different datasets at http://scikit-learn.org/stable/tutorial/machine_learning_map/.
LOCATION: 794

7       LINEAR REGRESSION
LOCATION: 805

Using the Seinfeld TV sitcom series as our data, let’s start by plotting the two following variables, with season number as the x coordinate and the number of viewers per season (in millions) as the y coordinate.
LOCATION: 813

Table 11: Seinfeld dataset
LOCATION: 816

Figure 15: Seinfeld dataset plotted on a scatterplot
LOCATION: 817

Figure 16: Linear regression hyperplane
LOCATION: 824

a two-dimensional space, a hyperplane serves as a (flat) trendline,
LOCATION: 827

The goal of linear regression is to split the data in a way that minimizes the distance between the hyperplane and the observed values.
LOCATION: 829

Figure 17: Error is the distance between the hyperplane and the observed value
LOCATION: 833

The Slope
LOCATION: 835

As one variable increases, the other variable will increase by the average value denoted by the hyperplane.
LOCATION: 837

Figure 18: Using the slope/hyperplane to make a prediction
LOCATION: 841

Linear Regression Formula
LOCATION: 844

y = bx + a.
LOCATION: 846

“a” is the point where the hyperplane crosses the y-axis, known as the y-intercept
LOCATION: 847

“b” dictates the steepness of the slope and explains the relationship between x and y
LOCATION: 849

Calculation Example
LOCATION: 851

Table 12: Sample dataset
LOCATION: 855

  Where: Σ = Total sum Σx = Total sum of all x values (1 + 2 + 1 + 4 + 3 = 11) Σy = Total sum of all y values (3 + 4 + 2 + 7 + 5 = 21) Σxy = Total sum of x*y for each row (3 + 8 + 2 + 28 + 15 = 56) Σx2 = Total sum of x*x for each row (1 + 4 + 1 + 16 + 9 = 31) n = Total number of rows. In the case of this example, n is equal to 5.
LOCATION: 857

Insert the “a” and “b” values into the linear formula. y = bx + a y = 1.441x + 1.029
LOCATION: 871

Figure 19: y = 1.441x + 1.029 plotted on the scatterplot
LOCATION: 879

Multiple Linear Regression
LOCATION: 881

The y-intercept is still expressed as a, but now there are multiple independent variables (represented as x1, x2, x3, etc.) each with their own respective coefficient (b1, b2, b3, etc).
LOCATION: 884

Discrete Variables
LOCATION: 894

the output (dependent variable) of linear regression must be continuous in the form of a floating-point or integer
LOCATION: 894

the input (independent variables) can be continuous or categorical.
LOCATION: 895

For categorical variables, i.e. gender, these variables must be expressed numerically using one-hot encoding
LOCATION: 896

Variable Selection
LOCATION: 898

On the one hand, adding more variables helps to account for more potential factors that control patterns in the data.
LOCATION: 899

On the other hand, this rationale only holds if the variables are relevant and possess some correlation/linear relationship with the dependent variable.
LOCATION: 900

In multiple linear regression, not only are the independent variables potentially related to the dependent variable, but they are also potentially related to each other.
LOCATION: 903

Figure 20: Simple linear regression (above) and multiple linear regression (below)
LOCATION: 905

If a strong linear correlation exists between two independent variables, this can lead to a problem called multi-collinearity.
LOCATION: 907

When two independent variables are strongly correlated, they have a tendency to cancel each other out and provide the model with little to no unique information.
LOCATION: 908

example of two multi-collinear variables are liters of fuel consumed and liters of fuel in the tank to predict how far a jet plane will fly.
LOCATION: 909

in this case negatively correlated; as one variables increases, the other variable decreases and vice versa.
LOCATION: 910

To avoid multi-collinearity, we need to check the relationship between each combination of independent variables using a scatterplot, pairplot (a matrix of relationships between variables), or correlation score.
LOCATION: 913

Figure 21: Pairplot with three variables
LOCATION: 920

Figure 22: Heatmap with three variables
LOCATION: 927

We can also use a pairplot, heatmap or correlation score to check if the independent variables are correlated to the dependent variable (and therefore relevant to the prediction outcome).
LOCATION: 928

CHAPTER QUIZ
LOCATION: 934

Using multiple linear regression, your task is to create a model to predict the tip amount guests will leave the restaurant when paying for their meal.
LOCATION: 935

1)    The dependent variable for this model should be which variable? A)    size B)    total_bill and tip C)    total_bill D)    tip
LOCATION: 940

2)    From looking only at the data preview above, which variable(s) appear to have a linear relationship with total_bill? A)    smoker B)    total_bill and size C)    time D)    sex
LOCATION: 947

3)    It’s important for the independent variables to be strongly correlated with the dependent variable and one or more of the other independent variables. True or False?
LOCATION: 955

8       LOGISTIC REGRESSION
LOCATION: 969

linear regression is useful for quantifying relationships between variables to predict a continuous outcome. Total bill and size (number of guests) are both examples of continuous variables.
LOCATION: 973

However, what if we want to predict a categorical variable such as “new customer” or “returning customer”? Unlike linear regression, the dependent variable (y) is no longer a continuous variable (such as total tip) but rather a discrete categorical variable.
LOCATION: 975

Logistic regression is still a supervised learning technique but produces a qualitative prediction rather than a quantitative prediction. This algorithm is often used to predict two discrete classes, e.g., pregnant or not pregnant.
LOCATION: 978

Using the sigmoid function, logistic regression finds the probability of independent variables (X) producing a discrete dependent variable (y) such as “spam” or “non-spam.”
LOCATION: 983

Where: x = the independent variable you wish to transform e = Euler's constant, 2.718
LOCATION: 986

Figure 23: A sigmoid function used to classify data points
LOCATION: 989

The sigmoid function produces an S-shaped curve that can convert any number and map it into a numerical value between 0 and 1 but without ever reaching those exact limits. Applying this formula, the sigmoid function converts independent variables into an expression of probability between 0 and 1 in relation to the dependent variable.
LOCATION: 991

Based on the found probabilities of the independent variables, logistic regression assigns each data point to a discrete class.
LOCATION: 996

Figure 24: An example of logistic regression
LOCATION: 1003

Although logistic regression shares a visual resemblance to linear regression, the logistic hyperplane represents a classification/decision boundary rather than a prediction trendline.
LOCATION: 1007

instead of using the hyperplane to make numeric predictions, the hyperplane is used to divide the dataset into classes.
LOCATION: 1008

The other distinction between logistic and linear regression is that the dependent variable (y) isn’t placed along the y-axis in logistic regression. Instead, independent variables can be plotted along both axes, and the class (output) of the dependent variable is determined by the position of the data point in relation to the decision boundary.
LOCATION: 1009

For classification scenarios with more than two possible discrete outcomes, multinomial logistic regression can be used
LOCATION: 1013

Figure 25: An example of multinomial logistic regression
LOCATION: 1016

Two tips to remember when using logistic regression are that the dataset should be free of missing values and that all independent variables are independent and not strongly correlated with each other.
LOCATION: 1022

Statistics 101: Logistic Regression series on YouTube by Brandon Foltz.[18]
LOCATION: 1028

CHAPTER QUIZ
LOCATION: 1030

1)    Which three variables (in their current form) could we use as the dependent variable to classify penguins?   2)    Which row(s) contains missing values?   3)    Which variable in the dataset preview is binary?
LOCATION: 1036

9       k-NEAREST NEIGHBORS
LOCATION: 1054

k-NN classifies new data points based on their position to nearby data points.
LOCATION: 1060

k-NN is similar to a voting system
LOCATION: 1061

Figure 26: An example of k-NN clustering used to predict the class of a new data point
LOCATION: 1067

set “k” to determine how many data points we want to use to classify the new data
LOCATION: 1071

If we set k to 3, k-NN analyzes the new data point’s position with respect to the three nearest data points (neighbors).
LOCATION: 1073

The outcome of selecting the three closest neighbors returns two Class B data points and one Class A data point.
LOCATION: 1074

It’s therefore useful to test numerous k combinations to find the best fit and avoid setting k too low or too high.
LOCATION: 1080

Setting k too low will increase bias and lead to misclassification
LOCATION: 1081

setting k too high will make it computationally expensive.
LOCATION: 1082

Five is the default number of neighbors for this algorithm using Scikit-learn.
LOCATION: 1085

this algorithm works best with continuous variables.
LOCATION: 1090

While k-NN is generally accurate and easy to comprehend, storing an entire dataset and calculating the distance between each new data point and all existing data points puts a heavy burden on computing resources.
LOCATION: 1101

NN is generally not recommended for analyzing large datasets.
LOCATION: 1105

Another downside is that it can be challenging to apply k-NN to high-dimensional data with a high number of features.
LOCATION: 1106

classify penguins into different species using the k-nearest neighbors algorithm, with k set to 5 (neighbors).
LOCATION: 1110

1)    Which of the following variables should we consider removing from our k-NN model? A. sex B. species C. body_mass_g D. bill_depth_mm
LOCATION: 1115

2)    If we wanted to reduce the processing time of our model, which of the following methods is recommended? A.    Increase k from 5 to 10 B.     Reduce k from 10 to 5 C.     Re-run the model and hope for a faster result D.    Increase the size of the training data
LOCATION: 1122

3)    To include the variable ‘sex’ in our model, which data scrubbing technique do we need to use?
LOCATION: 1131

ANSWERS   1)    A, sex (Binary variables should only be used when critical to the model’s accuracy.)   2)    B, Reduce k from 10 to 5   3)    One-hot encoding (to convert the variable into a numerical identifier of 0 or 1)
LOCATION: 1133

10       k-MEANS CLUSTERING
LOCATION: 1144

grouping or clustering data points that share similar attributes using unsupervised learning.
LOCATION: 1149

An online business, for example, wants to examine a segment of customers that purchase at the same time of the year and discern what factors influence their purchasing behavior.
LOCATION: 1150

As an unsupervised learning algorithm, k-means clustering attempts to divide data into k number of discrete groups and is highly effective at uncovering new patterns.
LOCATION: 1155

Figure 29: Comparison of original data and clustered data using k-means
LOCATION: 1159

splitting data into k number of clusters, with k representing the number of clusters you wish to create.
LOCATION: 1162

examine the unclustered data and manually select a centroid for each cluster. That centroid then forms the epicenter of an individual cluster.
LOCATION: 1169

The remaining data points on the scatterplot are then assigned to the nearest centroid by measuring the Euclidean distance.
LOCATION: 1173

Figure 30: Calculating Euclidean distance
LOCATION: 1175

Each data point can be assigned to only one cluster, and each cluster is discrete.
LOCATION: 1176

no overlap between clusters and no case of nesting a cluster inside another cluster.
LOCATION: 1177

After all data points have been allocated to a centroid, the next step is to aggregate the mean value of the data points in each cluster, which can be found by calculating the average x and y values of the data points contained in each cluster.
LOCATION: 1180

Next, take the mean value of the data points in each cluster and plug in those x and y values to update your centroid coordinates. This will most likely result in one or more changes to the location of your centroid(s).
LOCATION: 1182

Like musical chairs, the remaining data points rush to the closest centroid to form k number of clusters.
LOCATION: 1184

Should any data point on the scatterplot switch clusters with the changing of centroids, the previous step is repeated.
LOCATION: 1186

Figure 31: Sample data points are plotted on a scatterplot
LOCATION: 1192

Figure 32: Two existing data points are nominated as the centroids
LOCATION: 1194

Figure 33: Two clusters are formed after calculating the Euclidean distance of the remaining data points to the centroids.
LOCATION: 1196

Figure 34: The centroid coordinates for each cluster are updated to reflect the cluster’s mean value. The two previous centroids stay in their original position and two new centroids are added to the scatterplot. Lastly, as one data point has switched from the right cluster to the left cluster, the centroids of both clusters need to be updated one last time.
LOCATION: 1199

Figure 35: Two final clusters are produced based on the updated centroids for each cluster
LOCATION: 1204

Setting k
LOCATION: 1211

In general, as k increases, clusters become smaller and variance falls.
LOCATION: 1214

However, the downside is that neighboring clusters become less distinct from one another as k increases.
LOCATION: 1215

If you set k to the same number of data points in your dataset, each data point automatically becomes a standalone cluster.
LOCATION: 1216

In order to optimize k, you may wish to use a scree plot for guidance.
LOCATION: 1220

scree plot charts the degree of scattering (variance) inside a cluster as the total number of clusters increases.
LOCATION: 1221

A scree plot compares the Sum of Squared Error (SSE) for each variation of total clusters.
LOCATION: 1223

Figure 36: A scree plot
LOCATION: 1226

In general, you should opt for a cluster solution where SSE subsides dramatically to the left on the scree plot but before it reaches a point of negligible change with cluster variations to its right.
LOCATION: 1228

Another useful technique to decide the number of cluster solutions is to divide the total number of data points (n) by two and finding the square root.
LOCATION: 1233

A more simple and non-mathematical approach to setting k is to apply domain knowledge.
LOCATION: 1237

if I am analyzing data about visitors to the website
LOCATION: 1239

Because I already know there is a significant discrepancy in spending behavior between returning visitors and new visitors.
LOCATION: 1240

but understand that the effectiveness of “domain knowledge” diminishes dramatically past a low number of k clusters.
LOCATION: 1263

domain knowledge might be sufficient for determining two to four clusters but less valuable when choosing between a higher number of clusters, such as 20 or 21 clusters.
LOCATION: 1264

CHAPTER QUIZ   Your task is to group the flights dataset (which tracks flights from 1949 to 1960) into discrete clusters using k-means clustering.
LOCATION: 1266

1)    Using k-means clustering to analyze all 3 variables, what might be a good initial number of k clusters (using only domain/general knowledge) to train the model? k = 2 k = 100 k = 12 k = 3
LOCATION: 1271

What mathematical technique might we use to find the appropriate number of clusters? A.    Big elbow method B.     Mean absolute error C.     Scree plot D.    One-hot encoding
LOCATION: 1281

Which variable requires data scrubbing?
LOCATION: 1288

11       BIAS & VARIANCE
LOCATION: 1300

most algorithms have many different hyperparameters also leads to a vast number of potential outcomes.
LOCATION: 1306

Figure 37: Example of hyperparameters in Python for the algorithm gradient boosting
LOCATION: 1311

A constant challenge in machine learning is navigating underfitting and overfitting, which describe how closely your model follows the actual patterns of the data.
LOCATION: 1312

Bias refers to the gap between the value predicted by your model and the actual value of the data.
LOCATION: 1316

In the case of high bias, your predictions are likely to be skewed in a particular direction away from the true values.
LOCATION: 1317

Variance describes how scattered your predicted values are in relation to each other.
LOCATION: 1318

Figure 38: Shooting targets used to represent bias and variance
LOCATION: 1320

Ideally, you want a situation where there’s both low variance and low bias. In reality, however, there’s a trade-off between optimal bias and optimal variance.
LOCATION: 1335

Bias and variance both contribute to error but it’s the prediction error that you want to minimize, not the bias or variance specifically.
LOCATION: 1336

Peddling algorithms through the data is the easy part; the hard part is navigating bias and variance while maintaining a state of balance in your model.
LOCATION: 1338

Figure 39: Model complexity based on the prediction error
LOCATION: 1341

In Figure 39, we can see two curves.
LOCATION: 1342

The upper curve represents the test data, and the lower curve depicts the training data.
LOCATION: 1343

From the left, both curves begin at a point of high prediction error due to low variance and high bias. As they move toward the right, they change to the opposite: high variance and low bias.
LOCATION: 1343

Figure 40: Underfitting on the left and overfitting on the right
LOCATION: 1348

the model being overly simple and inflexible (underfitting) or overly complex and flexible (overfitting).
LOCATION: 1350

Underfitting (low variance, high bias) on the left and overfitting (high variance, low bias) on the right are shown in these two scatterplots.
LOCATION: 1352

A natural temptation is to add complexity to the model (as shown on the right) to improve accuracy, but this can, in turn, lead to overfitting.
LOCATION: 1352

Underfitting is when your model is overly simple, and again, has not scratched the surface of the underlying patterns in the data.
LOCATION: 1356

An advanced strategy to combat overfitting is to introduce regularization, which reduces the risk of overfitting by constraining the model to make it simpler.
LOCATION: 1368

one other technique to improve model accuracy is to perform cross validation, as covered earlier in Chapter 6, to minimize pattern discrepancies between the training data and the test data.
LOCATION: 1374

12       SUPPORT VECTOR MACHINES
LOCATION: 1378

SVM is mostly used as a classification technique for predicting categorical outcomes.
LOCATION: 1384

SVM is similar to logistic regression, in that it’s used to filter data into a binary or multiclass target variable.
LOCATION: 1385

Figure 41: Logistic regression versus SVM
LOCATION: 1388

gray zone that denotes margin, which is the distance between the decision boundary and the nearest data point, multiplied by two.
LOCATION: 1393

Figure 42: A new data point is added to the scatterplot
LOCATION: 1398

The new data point is a circle, but it’s located incorrectly on the left side of the logistic (A) decision boundary
LOCATION: 1399

The new data point, though, remains correctly located on the right side of the SVM (B) decision boundary (designated for circles) courtesy of ample “support” supplied by the margin.
LOCATION: 1400

Figure 43: Mitigating anomalies
LOCATION: 1403

A limitation of standard logistic regression is that it goes out of its way to fit outliers and anomalies
LOCATION: 1405

SVM, however, is less sensitive to such data points and actually minimizes their impact on the final location of the boundary line.
LOCATION: 1406

The SVM boundary can also be modified to ignore misclassified cases in the training data using a hyperparameter called C.
LOCATION: 1408

There is therefore a trade-off in SVM between a wide margin/more mistakes and a narrow margin/fewer mistakes.
LOCATION: 1412

Adding flexibility to the model using the hyperparameter C introduces what’s called a “soft margin,” which ignores a determined portion of cases that cross over the soft margin—leading to greater generalization in the model.
LOCATION: 1416

Figure 44: Soft margin versus hard margin
LOCATION: 1425

SVM’s real strength lies with high-dimensional data and handling multiple features.
LOCATION: 1427

SVM has numerous advanced variations available to classify high-dimensional data using what’s called the Kernel Trick.
LOCATION: 1428

Figure 45: In this example, the decision boundary provides a non-linear separator between the data in a 2-D space but transforms into a linear separator between data points when projected into a 3-D space
LOCATION: 1434

with a low feature-to-row ratio (low number of features relative to rows) due to speed and performance constraints.
LOCATION: 1442

SVM does, though, excel at untangling outliers from complex small and medium-sized datasets and managing high-dimensional data.
LOCATION: 1443

CHAPTER QUIZ   Using an SVM classifier, your task is to classify which island a penguin has come from after arriving on your own island.
LOCATION: 1444

Which of the following variables would be the dependent variable for this model? A. island B. species C. sex D. body_mass_g
LOCATION: 1450

2)    Which of the following variables could we use as independent variable(s)? A. island B. All of the variables C. All of the variables except island D. species
LOCATION: 1455

3)    What are two data scrubbing techniques commonly used with this algorithm?
LOCATION: 1460

13       ARTIFICIAL NEURAL NETWORKS
LOCATION: 1472

analyzing data through a network of decision layers.
LOCATION: 1480

The naming of this technique was inspired by the algorithm’s structural resemblance to the human brain.
LOCATION: 1480

Figure 46: Anatomy of a human brain neuron
LOCATION: 1483

artificial neural networks consist of interconnected decision functions, known as nodes, which interact with each other through axon-like edges.
LOCATION: 1486

Figure 47: The nodes, edges/weights, and sum/activation function of a basic neural network
LOCATION: 1492

Each edge in the network has a numeric weight that can be altered based on experience.
LOCATION: 1494

the sum of the connected edges satisfies a set threshold, known as the activation function, this activates a neuron at the next layer.
LOCATION: 1495

Using supervised learning, the model’s predicted output is compared to the actual output (that’s known to be correct), and the difference between these two results is measured as the cost or cost value.
LOCATION: 1499

The purpose of training is to reduce the cost value until the model’s prediction closely matches the correct output.
LOCATION: 1501

This is achieved by incrementally tweaking the network’s weights until the lowest possible cost value is obtained.
LOCATION: 1502

This particular process of training the neural network is called back-propagation.
LOCATION: 1503

The Black-box Dilemma
LOCATION: 1506

Although the network can approximate accurate outputs, tracing its decision structure reveals limited to no insight into how specific variables influence its decision.
LOCATION: 1508

For instance, if we use a neural network to predict the outcome of a Kickstarter campaign (an online funding platform for creative projects), the network can analyze numerous independent variables including campaign category, currency, deadline, and minimum pledge amount, etc.
LOCATION: 1509

However, the model is unable to specify the relationship of these independent variables to the dependent variable of the campaign reaching its funding target.
LOCATION: 1511

Moreover, it’s possible for two neural networks with different topologies and weights to produce the same output, which makes it even more challenging to trace the impact of specific variables
LOCATION: 1513

neural networks generally fit prediction tasks with a large number of input features and complex patterns,
LOCATION: 1516

especially problems that are difficult for computers to decipher but simple and almost trivial for humans.
LOCATION: 1516

One example is the CAPTCHA
LOCATION: 1517

Another example is identifying if a pedestrian is preparing to step into the path of an oncoming vehicle.
LOCATION: 1519

In both examples, obtaining a fast and accurate prediction is more important than decoding the specific variables and their relationship to the final output.
LOCATION: 1520

Building a Neural Network
LOCATION: 1522

A typical neural network can be divided into input, hidden, and output layers.
LOCATION: 1522

Figure 48: The three general layers of a neural network
LOCATION: 1526

While there are many techniques to assemble the nodes of a neural network, the simplest method is the feed-forward network where signals flow only in one direction and there’s no loop in the network.
LOCATION: 1534

The most basic form of a feed-forward neural network is the perceptron,
LOCATION: 1535

Figure 50: Weights are added to the perceptron
LOCATION: 1561

Next, we multiply each weight by its input: Input 1: 24 * 0.5 = 12 Input 2: 16 * -1 = -16
LOCATION: 1564

Thus: Input 1: 24 * 0.5 = 12 Input 2: 16 * -1.0 = -16 Sum (Σ): 12 + -16 = -4
LOCATION: 1573

As a numeric value less than zero, the result produces “0” and does not trigger the perceptron’s activation function.
LOCATION: 1576

Input 1: 24 * 0.5 = 12 Input 2: 16 * -0.5 = -8 Sum (Σ): 12 + -8 = 4
LOCATION: 1579

As a positive outcome, the perceptron now produces “1” which triggers the activation function,
LOCATION: 1582

A weakness of a perceptron is that because the output is binary (0 or 1), small changes in the weights or bias in any single perceptron within a larger neural network can induce polarizing results.
LOCATION: 1589

An alternative to the perceptron is the sigmoid neuron.
LOCATION: 1592

similar to a perceptron, but the presence of a sigmoid function rather than a binary filter
LOCATION: 1593

While more flexible than a perceptron, a sigmoid neuron is unable to generate negative values.
LOCATION: 1596

third option is the hyperbolic tangent function.
LOCATION: 1597

Figure 53: A hyperbolic tangent function graph
LOCATION: 1599

Multilayer Perceptrons
LOCATION: 1603

algorithm for predicting a categorical (classification) or continuous (regression) target variable.
LOCATION: 1604

powerful because they aggregate multiple models into a unified prediction model,
LOCATION: 1605

Figure 54: A multilayer perceptron used to classify a social media user’s political preference
LOCATION: 1607

multilayer perceptrons are ideal for interpreting large and complex datasets with no time or computational restraints.
LOCATION: 1616

Less compute-intensive algorithms, such as decision trees and logistic regression, for example, are more efficient for working with smaller datasets.
LOCATION: 1617

Deep Learning
LOCATION: 1621

as patterns in the data become more complicated—especially
LOCATION: 1623

a shallow model is no longer reliable or capable of sophisticated analysis because the model becomes exponentially complicated as the number of inputs increases.
LOCATION: 1624

A neural network, with a deep number of layers, though, can be used to interpret a high number of input features and break down complex patterns into simpler patterns, as shown in Figure 55.
LOCATION: 1625

Figure 55: Facial recognition using deep learning. Source: kdnuggets.com
LOCATION: 1627

What makes deep learning “deep” is the stacking of at least 5-10 node layers.
LOCATION: 1632

Object recognition, as used by self-driving cars to recognize objects such as pedestrians and other vehicles, uses upward of 150 layers and is a popular application of deep learning.
LOCATION: 1633

Table 13: Common usage scenarios and paired deep learning techniques
LOCATION: 1639

multilayer perceptrons (MLP) have largely been superseded by new deep learning techniques such as convolution networks, recurrent networks, deep belief networks, and recursive neural tensor networks (RNTN).
LOCATION: 1640

CHAPTER QUIZ   Using a multilayer perceptron, your job is to create a model to classify the gender sex) of penguins that have been affected and rescued during a natural disaster. However, you can only use the physical attributes of penguins to train your model.
LOCATION: 1644

1)    How many output nodes does the multilayer perceptron need to predict the dependent variable of sex (gender)?
LOCATION: 1650

2)    Which of the seven variables could we use as independent variables based on only the penguin’s physical attributes?
LOCATION: 1653

3)    Which is a more transparent classification algorithm that we could use in replace of a multilayer perceptron? A.     Simple linear regression B.     Logistic regression C.     k-means clustering D.     Multiple linear regression
LOCATION: 1656

14       DECISION TREES
LOCATION: 1673

The huge amount of input data and computational resources required to train a neural network is the first downside
LOCATION: 1681

other major downside of neural networks is the black-box dilemma, which conceals the model’s decision structure.
LOCATION: 1684

Decision trees, on the other hand, are transparent and easy to interpret. They work with less data and consume less computational resources.
LOCATION: 1685

Decision trees are used primarily for solving classification problems but can also be used as a regression model to predict numeric outcomes.
LOCATION: 1687

Figure 56: Example of a regression tree
LOCATION: 1692

Figure 57: Example of a classification tree
LOCATION: 1694

Building a Decision Tree
LOCATION: 1698

Decision trees start with a root node that acts as a starting point and is followed by splits that produce branches, also known as edges.
LOCATION: 1698

The branches then link to leaves, also known as nodes, which form decision points.
LOCATION: 1700

The aim is to keep the tree as small as possible. This is achieved by selecting a variable that optimally splits the data into homogenous groups,
LOCATION: 1704

we want the data at each layer to be more homogenous than the previous partition.
LOCATION: 1707

We therefore want to pick a “greedy” algorithm that can reduce entropy at each layer of the tree. An example of a greedy algorithm is the Iterative Dichotomizer (ID3), invented by J.R. Quinlan.
LOCATION: 1707

Table 14: Employee characteristics
LOCATION: 1712

Let’s first split the data by variable 1
LOCATION: 1716

Black = Promoted, White = Not Promoted
LOCATION: 1721

Now let’s try variable 2
LOCATION: 1722

Black = Promoted, White = Not Promoted
LOCATION: 1732

Lastly, we have variable 3
LOCATION: 1733

Black = Promoted, White = Not Promoted
LOCATION: 1742

Calculating Entropy
LOCATION: 1752

(-p1logp1 - p2logp2) / log2
LOCATION: 1761

Overfitting
LOCATION: 1818

The variable used to first split the data does not guarantee the most accurate model at the end of production.
LOCATION: 1826

Thus, although decision trees are highly visual and effective at classifying a single set of data, they are also inflexible and vulnerable to overfitting, especially for datasets with high pattern variance.
LOCATION: 1828

Bagging
LOCATION: 1830

involves growing multiple decision trees using a randomized selection of input data for each tree and combining the results by averaging the output (for regression) or voting (for classification).
LOCATION: 1832

Random Forests
LOCATION: 1841

artificially limit the choice of variables by capping the number of variables considered for each split.
LOCATION: 1843

Figure 58: Example of growing random trees to produce a prediction
LOCATION: 1852

Scott Hartshorn advises focusing on optimizing other hyperparameters before adding more trees to the initial model, as this will reduce processing time in the short term and increasing the number of trees later should
LOCATION: 1856

other techniques including gradient boosting tend to return superior prediction accuracy.
LOCATION: 1859

Random forests, though, are fast to train and work well for obtaining a quick benchmark model.
LOCATION: 1860

Boosting
LOCATION: 1861

combining “weak” models into one “strong” model.
LOCATION: 1863

achieved by adding weights to trees based on misclassified cases in the previous tree.
LOCATION: 1865

One of the more popular boosting algorithms is gradient boosting.
LOCATION: 1868

Rather than selecting combinations of variables at random, gradient boosting selects variables that improve prediction accuracy with each new tree.
LOCATION: 1869

Figure 59: Example of reducing prediction error across multiple trees to produce a prediction
LOCATION: 1876

Boosting also mitigates the issue of overfitting and it does so using fewer trees than random forests.
LOCATION: 1878

While adding more trees to a random forest usually helps to offset overfitting, the same process can cause overfitting in the case of boosting
LOCATION: 1879

overfitting can be explained by their highly-tuned focus on learning and reiterating from earlier mistakes.
LOCATION: 1881

it can lead to mixed results in the case of data stretched by a high number of outliers.
LOCATION: 1882

The other main downside of boosting is the slow processing speed that comes with training a sequential decision model.
LOCATION: 1885

CHAPTER QUIZ   Your task is to predict the body mass (body_mass_g) of penguins using the penguin dataset and the random forests algorithm.
LOCATION: 1894

1) Which variables could we use as independent variables to train our model?
LOCATION: 1899

2) To train a quick benchmark model, gradient boosting is faster to train than random forests. True or False?
LOCATION: 1901

3) Which tree-based technique can be easily visualized? A. Decision trees B. Gradient boosting C. Random forests
LOCATION: 1903

ANSWERS
LOCATION: 1908

1)    All variables except for body_mass_g (Tree-based techniques work well with both discrete and continuous variables as input variables.)
LOCATION: 1909

2)    False (Gradient boosting runs sequentially, making it slower to train. A random forest is trained simultaneously, making it faster to train.)
LOCATION: 1913

3)    A, Decision trees
LOCATION: 1916

15       ENSEMBLE MODELING
LOCATION: 1920

By combining the output of different models (instead of relying on a single estimate), ensemble modeling helps to build a consensus on the meaning of the data.
LOCATION: 1929

In the case of classification, multiple models are consolidated into a single prediction using a voting system[25]
LOCATION: 1932

can also be divided into sequential or parallel and homogenous or heterogeneous.
LOCATION: 1936

sequential and parallel models.
LOCATION: 1937

In the case of the former, the model’s prediction error is reduced by adding weights to classifiers that previously misclassified data.
LOCATION: 1937

Gradient boosting and AdaBoost (designed for classification problems) are both examples of sequential models.
LOCATION: 1938

parallel ensemble models work concurrently and reduce error by averaging.
LOCATION: 1939

Random forests are an example of this technique.
LOCATION: 1939

Ensemble models can be generated using a single technique with numerous variations, known as a homogeneous ensemble, or through different techniques, known as a heterogeneous ensemble.
LOCATION: 1940

An example of a homogeneous ensemble model would be multiple decision trees working together to form a single prediction (i.e. bagging).
LOCATION: 1941

an example of a heterogeneous ensemble would be the usage of k-means clustering or a neural network in collaboration with a decision tree algorithm.
LOCATION: 1942

it’s important to select techniques that complement each other.
LOCATION: 1944

Neural networks, for instance, require complete data for analysis, whereas decision trees are competent at handling missing values.[28]
LOCATION: 1945

Together, these two techniques provide added benefit over a homogeneous model.
LOCATION: 1947

there are four main methods: bagging, boosting, a bucket of models, and stacking.
LOCATION: 1955

bucket of models trains multiple different algorithmic models using the same training data and then picks the one that performed most accurately on the test data.
LOCATION: 1956

Bagging, as we know, is an example of parallel model averaging using a homogenous ensemble, which draws upon randomly drawn data and combines predictions to design a unified model.
LOCATION: 1958

Boosting is a popular alternative technique that is still a homogenous ensemble but addresses errors and data misclassified by the previous iteration to produce a sequential model.
LOCATION: 1960

Stacking runs multiple models simultaneously on the data and combines those results to produce a final model.
LOCATION: 1963

Unlike boosting and bagging, stacking usually combines outputs from different algorithms (heterogenous) rather than altering the hyperparameters of the same algorithm (homogenous).
LOCATION: 1964

Figure 60: Stacking algorithm
LOCATION: 1969

the gains of using a stacking technique are marginal in line with the level of complexity, and organizations usually opt for the ease and efficiency of boosting or bagging.
LOCATION: 1970

The Netflix Prize competition, held between 2006 and 2009, offered a prize for a machine learning model that could significantly improve Netflix’s content recommender system.
LOCATION: 1972

One of the winning techniques, from the team BellKor’s Pragmatic Chaos, adopted a form of linear stacking that blended predictions from hundreds of different models using different algorithms.
LOCATION: 1973

16       DEVELOPMENT ENVIRONMENT
LOCATION: 1978

(http://jupyter.org/install.html)
LOCATION: 1991

(https://www.anaconda.com/products/individual/).
LOCATION: 1993

The Melbourne_housing_FULL dataset can be downloaded from this link: https://www.kaggle.com/anthonypino/melbourne-housing-market/.
LOCATION: 2019

df = pd.read_csv('~/Downloads/Melbourne_housing_FULL.csv')
LOCATION: 2025

df.head()
LOCATION: 2041

Figure 68: Finding a row using .iloc[ ]
LOCATION: 2064

df.columns   Figure 69: Print columns
LOCATION: 2073

17       BUILDING A MODEL IN PYTHON
LOCATION: 2080

#Import libraries import pandas as pd from sklearn.model_selection import train_test_split from sklearn import ensemble from sklearn.metrics import mean_absolute_error
LOCATION: 2098

df = pd.read_csv('~/Downloads/Melbourne_housing_FULL.csv')
LOCATION: 2107

# The misspellings of “longitude” and “latitude” are preserved here del df['Address'] del df['Method'] del df['SellerG'] del df['Date'] del df['Postcode'] del df['Lattitude'] del df['Longtitude'] del df['Regionname'] del df['Propertycount']
LOCATION: 2130

The remaining eleven independent variables from the dataset are Suburb, Rooms, Type, Distance, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt, and CouncilArea.
LOCATION: 2141

The twelfth variable is the dependent variable which is Price.
LOCATION: 2142

df.dropna(axis = 0, how = 'any', subset = None, inplace = True)   Table 16: Dropna parameters
LOCATION: 2154

df = pd.get_dummies(df, columns = ['Suburb', 'CouncilArea', 'Type'])
LOCATION: 2169

X = df.drop('Price',axis=1) y = df['Price']
LOCATION: 2175

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, shuffle = True)
LOCATION: 2181

assign our chosen algorithm (gradient boosting regressor) as a new variable (model) and configure its hyperparameters
LOCATION: 2183

model = ensemble.GradientBoostingRegressor(     n_estimators = 150,     learning_rate = 0.1,     max_depth = 30,     min_samples_split = 4,     min_samples_leaf = 6,     max_features = 0.6,     loss = 'huber' )
LOCATION: 2185

The first line is the algorithm itself (gradient boosting)
LOCATION: 2190

n_estimators states the number of decision trees.
LOCATION: 2192

a high number of trees generally improves accuracy (up to a certain point) but will inevitably extend the model’s processing time.
LOCATION: 2193

learning_rate controls the rate at which additional decision trees influence the overall prediction.
LOCATION: 2195

Inserting a low rate here, such as 0.1, should help to improve accuracy.
LOCATION: 2197

max_depth defines the maximum number of layers (depth) for each decision tree.
LOCATION: 2198

If “None” is selected, then nodes expand until all leaves are pure or until all leaves contain less than min_samples_leaf.
LOCATION: 2199

min_samples_split defines the minimum number of samples required to execute a new binary split.
LOCATION: 2203

min_samples_leaf represents the minimum number of samples that must appear in each child node (leaf) before a new branch can be implemented.
LOCATION: 2206

max_features is the total number of features presented to the model when determining the best split.
LOCATION: 2210

loss calculates the model's error rate.
LOCATION: 2218

we are using huber which protects against outliers and anomalies.
LOCATION: 2219

Alternative error rate options include ls (least squares regression), lad (least absolute deviations), and quantile (quantile regression).
LOCATION: 2220

fit() function from Scikit-learn to link the training data to the learning algorithm stored in the variable model to train the prediction model.
LOCATION: 2228

model.fit(X_train, y_train)
LOCATION: 2231

predict() function from Scikit-learn to run the model on the X_train data and evaluate its performance against the actual y_train data.
LOCATION: 2234

mae_train = mean_absolute_error(y_train, model.predict(X_train)) print ("Training Set Mean Absolute Error: %.2f" % mae_train)
LOCATION: 2239

mae_test = mean_absolute_error(y_test, model.predict(X_test)) print ("Test Set Mean Absolute Error: %.2f" % mae_test)
LOCATION: 2251

For this model, our training set’s mean absolute error is $27,256.70, and the test set’s mean absolute error is $166,103.04.
LOCATION: 2257

While $27,256.70 may seem like a lot of money, this average error value is low given the maximum range of our dataset is $8 million.
LOCATION: 2261

A high discrepancy between the training and test data is usually an indicator of overfitting in the model.
LOCATION: 2264

mini course at https://scatterplotpress.com/p/house-prediction-model.
LOCATION: 2274

18       MODEL OPTIMIZATION
LOCATION: 2277

want to improve its prediction accuracy with future data and reduce the effects of overfitting.
LOCATION: 2282

starting point is to modify the model’s hyperparameters.
LOCATION: 2283

Holding the other hyperparameters constant, let’s begin by adjusting the maximum depth from “30” to “5.”
LOCATION: 2283

Although the mean absolute error of the training set is now higher, this helps to reduce the issue of overfitting and should improve the model’s performance.
LOCATION: 2286

Another step to optimize the model is to add more trees. If we set n_estimators to 250,
LOCATION: 2288

This second optimization reduces the training set’s absolute error rate by approximately $10,000
LOCATION: 2292

While manual trial and error can be a useful technique to understand the impact of variable selection and hyperparameters, there are also automated techniques for model optimization, such as grid search.
LOCATION: 2306

Grid search allows you to list a range of configurations you wish to test for each hyperparameter and methodically test each
LOCATION: 2308

grid search does take a long time to run![33] It sometimes helps to run a relatively coarse grid search using consecutive powers of 10 (i.e. 0.01, 0.1, 1, 10) and then run a finer grid search around the best value identified.[34]
LOCATION: 2311

Another way of optimizing algorithm hyperparameters is the randomized search method using Scikit-learn’s RandomizedSearchCV.
LOCATION: 2315

more advanced tutorial available at https://scatterplotpress.com/p/house-prediction-model.
LOCATION: 2324

Code for the Optimized Model
LOCATION: 2328

Code for Grid Search Model
LOCATION: 2357

NEXT STEPS   6 Video Tutorials
LOCATION: 2391

six free video tutorials at https://scatterplotpress.com/p/ml-code-exercises.
LOCATION: 2407

Building a House Prediction Model in Python
LOCATION: 2410

this free chapter in video format at https://scatterplotpress.com/p/house-prediction-model.
LOCATION: 2414

FURTHER RESOURCES
LOCATION: 2448

Machine Learning Format: Free Coursera course Presenter: Andrew Ng
LOCATION: 2453

Project 3: Reinforcement Learning Format: Online blog tutorial Author: EECS Berkeley
LOCATION: 2460

Basic Algorithms
LOCATION: 2465

Machine Learning With Random Forests And Decision Trees: A Visual Guide For Beginners  Format: E-book Author: Scott Hartshorn
LOCATION: 2468

Linear Regression And Correlation: A Beginner's Guide Format: E-book Author: Scott Hartshorn
LOCATION: 2475

The Future of AI
LOCATION: 2480

The Inevitable: Understanding the 12 Technological Forces That Will Shape Our Future Format: E-Book, Book, Audiobook Author: Kevin Kelly
LOCATION: 2483

Homo Deus: A Brief History of Tomorrow Format: E-Book, Book, Audiobook Author: Yuval Noah Harari
LOCATION: 2491

Programming
LOCATION: 2498

Learning Python, 5th Edition Format: E-Book, Book Author: Mark Lutz
LOCATION: 2500

Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems Format: E-Book, Book Author: Aurélien Géron
LOCATION: 2507

Recommender Systems
LOCATION: 2514

The Netflix Prize and Production Machine Learning Systems: An Insider Look Format: Blog Author: Mathworks
LOCATION: 2516

Recommender Systems Format: Coursera course Presenter: The University of Minnesota
LOCATION: 2522

Deep Learning Simplified Format: Blog Channel: DeepLearning.TV
LOCATION: 2531

Deep Learning Specialization: Master Deep Learning, and Break into AI Format: Coursera course Presenter: deeplearning.ai and NVIDIA
LOCATION: 2536

Deep Learning Nanodegree Format: Udacity course Presenter: Udacity
LOCATION: 2543

Future Careers
LOCATION: 2551

Will a Robot Take My Job? Format: Online article Author: The BBC
LOCATION: 2552

So You Wanna Be a Data Scientist? A Guide to 2015's Hottest Profession Format: Blog Author: Todd Wasserman
LOCATION: 2557

OTHER BOOKS BY THE AUTHOR
LOCATION: 2725

AI for Absolute Beginners
LOCATION: 2728

Machine Learning with Python for Beginners
LOCATION: 2732

Machine Learning: Make Your Own Recommender System
LOCATION: 2735

Data Analytics for Absolute Beginners
LOCATION: 2739

Statistics for Absolute Beginners
LOCATION: 2742

Generative AI Art for Beginners
LOCATION: 2748

ChatGPT Prompts Book
LOCATION: 2751

[15] Standard deviation is a measure of spread among data points. It measures variability by calculating the average squared distance of all data observations from the mean of the dataset.
LOCATION: 2791

[18] Brandon Foltz, “Logistic Regression,” YouTube, https://www.youtube.com/channel/UCFrjdcImgcQVyFbK04MBEhA
LOCATION: 2802
