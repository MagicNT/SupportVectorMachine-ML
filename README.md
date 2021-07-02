# Support Vector Machine Algorithm

## **Implementation**

### Scripting Language

- Python 3


### Import Libraries

- We import the numpy Python library to utilize its vast libraries for creating and manipulating multi-dimensional arrays and matrices as well as high-level math functions to operate on these arrays
- We import the cvxopt Python library to utilize its convex optimization functions when dealing with matrix operations
- We import the prettytable Python library to present the output results in a nice tabular layout
- We import the sklearn Python library to utilize its metric calculation functions
- We import the matplotlib Python library to create interactive plot visualization for showcasing the resulting support vectors


```python
import numpy
import cvxopt
import prettytable
import cvxopt.solvers
import sklearn.metrics
import matplotlib.lines as matl
import matplotlib.pyplot as matp
```

### Plot Class

We implemented a module for handling all plotting operations whenever the algorithm has concluded and results are ready to be displayed. In the initialization function of the class we pass along as parameters the data; nagative points array, positive points array and the Support Vector Machine object originally created for which the algorithm relies on.


```python
class Plot:

    def __init__(self, X1, X2, SVM):
        self.SVM = SVM
        self.X1 = X1
        self.X2 = X2
        self.x1_min = X1.min()
        self.x1_max = X1.max()
        self.x2_min = X2.min()
        self.x2_max = X2.max()
```

### Prepare Plot Labels and Legend Description

In the below implemented functions we setup the final plot visualization axis labels along with the title. In addition, we setup the legend description for the plot so that the viewer can navigate the drawing. The final plot will display the positive/negative class data points, decision boundary, margins and the support vectors. 


```python
    def setup_labels(self):
        matp.xlabel("X1", fontsize=15)
        matp.ylabel("X2", fontsize=15)
        matp.title("SVM — Linear Kernel", fontsize=15);


    def setup_legend(self):
        legend_description = [
            "Negative Class",
            "Positive Class",
            "Decision Boundary",
            "Margins",
            "Support Vectors"
        ]
        legend = [
            matl.Line2D([0], [0], linestyle="none", marker="x", color="red", markerfacecolor="red", markersize=9),
            matl.Line2D([0], [0], linestyle="none", marker="o", color="green", markerfacecolor="green", markersize=9),
            matl.Line2D([0], [0], linestyle="-", marker=".", color="black", markerfacecolor="green",markersize=0),
            matl.Line2D([0], [0], linestyle="--", marker=".", color="black", markerfacecolor="green", markersize=0),
            matl.Line2D([0], [0], linestyle="none", marker=".", color="blue", markerfacecolor="blue", markersize=9)
        ]
        matp.legend(legend, legend_description, fontsize="7", loc="upper left").get_frame().set_linewidth(0.3)
```

### Drawing Data Points

In the below implemented routine we take the locally saved data variables and place them on the plot visualization by giving each class a corresponding marker (X, O) as well as color.


```python
    def draw_points(self):
        matp.plot(self.X1[:, 0], self.X1[:, 1], marker="x", markersize=5, color="red", linestyle="none")
        matp.plot(self.X2[:, 0], self.X2[:, 1], marker="o", markersize=4, color="green", linestyle="none")
        matp.scatter(self.SVM.support_vector[:, 0], self.SVM.support_vector[:, 1], s=60, color="blue")
```

### Drawing Margins and Decision Boundary

A linear discriminative classifier would attempt to draw a straight line separating the two sets of data, and thereby create a model for classification. The below implemented function along with its helper function draw the margin lines and the decision boundary on the final plot visualization. They take as parameters the data (x), the coefficient weights (w) and the soft marging parameter here being 0 to signify a hard margin.


```python
    def f(self, x, w, b, c=0):
        return (-w[0] * x - b + c) / w[1]


    def draw_margin_boundary(self, c, string):
        x = [
            self.x1_min,
            self.x1_max
        ]
        y = [
            self.f(self.x1_min, self.SVM.weight, self.SVM.intercept, c), 
            self.f(self.x1_max, self.SVM.weight, self.SVM.intercept, c)
        ]
        matp.plot(x, y, string)
```

### Make Plot

The below routine createa a figure object and sets up the plotting area then calls on the class functions to set the labels and the plot legend, place the data points on the plot provided through the local vector variables, then calls on the margin drawing function to generate and draw the decision bondary, upper margin and lower margin lines. 


```python
    def plot(self):
        fig = matp.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim=(self.x1_min, self.x1_max), ylim=(self.x2_min, self.x2_max))
        self.setup_labels()
        self.setup_legend()
        self.draw_points()
        self.draw_margin_boundary(0, "k")
        self.draw_margin_boundary(1, "k--")
        self.draw_margin_boundary(-1, "k--")
        matp.show()
```

### Synthetic Dataset

In this algorithm implementation we artificially generate a 2-dimensional linear data that can be seperated with a linear function (line). We utilize this dataset by following up and directly splitting it into testing data and training data through the specification of a percentage paramater that divides the data up. In the resulting process of this class's main function we save X and Y axis coordinates (1-dimensional) for each set; for testing and training.


```python
class Dataset:

    def __init__(self, seed=5, percent=0.8):
        self.seed = seed
        self.percent = percent
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None


    def linear_dataset(self):
        xy = []
        numpy.random.seed(self.seed)
        cov = numpy.array([[0.8, 0.6], [0.6, 0.8]])
        mean1 = numpy.array([0, 2])
        xy.append(numpy.random.multivariate_normal(mean1, cov, 100))
        xy.append(numpy.ones(len(xy[0])))
        mean2 = numpy.array([2, 0])
        xy.append(numpy.random.multivariate_normal(mean2, cov, 100))
        xy.append(numpy.ones(len(xy[2])) * -1)
        
        xy_test = []
        xy_train = []
        cutOff = int(len(xy[0])*self.percent);
        for e in xy:
            xy_train.append(e[:cutOff])
            xy_test.append(e[cutOff:])

        self.x_train = numpy.vstack((xy_train[0], xy_train[2]))
        self.y_train = numpy.hstack((xy_train[1], xy_train[3]))
        self.x_test = numpy.vstack((xy_test[0], xy_test[2]))
        self.y_test = numpy.hstack((xy_test[1], xy_test[3]))
```

### Support Vector Machine Algorithm Class

In this assignment question we will implement a Linear-Kernel Support Vector Machine algorithm. We implemented the algorithm in a modular form as a class for better management and manipulation control. It should be noted that the local class variables represent the lagrange multipliers, the support vectors, the intercept since this a linear kernel and the weight vector.


```python
class Supportvectormachine:

    def __init__(self):
        self.alphas = []
        self.support_vector = []
        self.sv_y = []
        self.intercept = 0
        self.weight = None
```


```python

```

### Decision Boundary

In this below routine we create the decision boundary for the plotting to be displayed later on. In addition we calculate the hypothesis. 


```python
    def linear_kernel(self, x1, x2):
        return numpy.dot(x1, x2)


    def predict(self, X):
        return numpy.sign(self.project(X))


    def project(self, X):
        if self.weight is not None:
            return numpy.dot(X, self.weight) + self.intercept
        else:
            y_predict = numpy.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.alphas, self.sv_y, self.support_vector):
                    s += a * sv_y * self.linear_kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.intercept
```

### Fitting Routine

In the below routine we write procedure to fit the training data and the predicted data. We begin by creating the Gram matrix and applying kernel trick. Afterwards, we initiate and setup the cvxopt library, to solve with its quadratic programming optimisation problem and obtain the lagrange multipliers which we then locally save in the alphas variable. It represents a vector made of flattened matrix of all the lagrange multipliers. Next we proceed to build the support vectors, the intercept and the weight vectors.


```python
    def fit(self, X, y):
        n_samples, n_features = X.shape

        K = numpy.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.linear_kernel(X[i], X[j])

        cvxopt.solvers.options["show_progress"] = False
        cvxopt.solvers.options["abstol"] = 1e-10
        cvxopt.solvers.options["reltol"] = 1e-10
        cvxopt.solvers.options["feastol"] = 1e-10
        alphas = numpy.ravel(cvxopt.solvers.qp(
            cvxopt.matrix(numpy.outer(y, y) * K),
            cvxopt.matrix(numpy.ones(n_samples) * -1),
            cvxopt.matrix(numpy.diag(numpy.ones(n_samples) * -1)),
            cvxopt.matrix(numpy.zeros(n_samples)),
            cvxopt.matrix(y, (1, n_samples)),
            cvxopt.matrix(0.0)
            )["x"]
        )

        sv = alphas > 1e-5
        ind = numpy.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.support_vector = X[sv]
        self.sv_y = y[sv]
        for n in range(len(self.alphas)):
            self.intercept += self.sv_y[n] - numpy.sum(self.alphas*self.sv_y*K[ind[n], sv])
        self.intercept = self.intercept/len(self.alphas)
        self.weight = numpy.zeros(n_features)
        for n in range(len(self.alphas)):
            self.weight += self.alphas[n] * self.sv_y[n] * self.support_vector[n]
```

### Driver Routine

Below is the main driver routine which creates a Dataset object with which we then generate a linear dataset that we will utilize for training and testing the SVM model, after splitting it depending on a percentage of our choice. Using this created model object, we create a SupportVectorMachine object and call the fitting routine on it to fit the training data which we obtained its X/Y vector from the Dataset object we previosuly created. Next, we fit the predicted data by calling the implemented in-class function predict. Next we display in a nice tabular form, how many predictions were made correctly as well as how many support vectors were found. Finally, we create a Plot object and pass to it the training data and the SVM object we created and fitted. Through this Plot object we made a plot visualization that shows us how the training datapoints are scattered and demonstrates the decision boundary, margins and support vectors.


```python
def main():
    print("\n" + "="*70)
    print(" Support Vector Machine Algrithm")
    print("="*70 + "\n")
    DS = Dataset()
    DS.linear_dataset()
    x_train, y_train, x_test, y_test = [DS.x_train, DS.y_train, DS.x_test, DS.y_test]
    SVM = Supportvectormachine()
    SVM.fit(x_train, y_train)
    y_predict = SVM.predict(x_test)
    correct_predictions = numpy.sum(y_predict == y_test)
    
    print(" [+] Support Vectors: \n")
    t = prettytable.PrettyTable(["# Support Vectors", "# Train Examples"])
    t.add_row([len(SVM.support_vector), len(x_train)])
    print(t)
    
    print("\n [+] Predictions: \n")
    t = prettytable.PrettyTable(["# Correct Predictions", "# Test Examples"])
    t.add_row([correct_predictions, len(x_test)])
    print(t)    

    print("\n [+] Evaluation Metric Scores: \n")
    t = prettytable.PrettyTable(["Metric", "Score"])
    t.add_row(["Accuracy", sklearn.metrics.accuracy_score(y_test, y_predict)])
    t.add_row(["F1", sklearn.metrics.f1_score(y_test, y_predict)])
    t.add_row(["Precision", sklearn.metrics.precision_score(y_test, y_predict)])
    t.add_row(["Recall", sklearn.metrics.recall_score(y_test, y_predict)])
    print(t)    

    Plot(x_train[y_train == 1], x_train[y_train == -1], SVM).plot()
```

### Full Code

Here is the full code put together.


```python
import numpy
import cvxopt
import prettytable
import cvxopt.solvers
import sklearn.metrics
import matplotlib.lines as matl
import matplotlib.pyplot as matp



class Plot:

    def __init__(self, X1, X2, SVM):
        self.SVM = SVM
        self.X1 = X1
        self.X2 = X2
        self.x1_min = X1.min()
        self.x1_max = X1.max()
        self.x2_min = X2.min()
        self.x2_max = X2.max()


    def setup_labels(self):
        matp.xlabel("X1", fontsize=15)
        matp.ylabel("X2", fontsize=15)
        matp.title("SVM — Linear Kernel", fontsize=15)


    def setup_legend(self):
        legend_description = [
            "Negative Class",
            "Positive Class",
            "Decision Boundary",
            "Margins",
            "Support Vectors"
        ]
        legend = [
            matl.Line2D([0], [0], linestyle="none", marker="x", color="red", markerfacecolor="red", markersize=9),
            matl.Line2D([0], [0], linestyle="none", marker="o", color="green", markerfacecolor="green", markersize=9),
            matl.Line2D([0], [0], linestyle="-", marker=".", color="black", markerfacecolor="green",markersize=0),
            matl.Line2D([0], [0], linestyle="--", marker=".", color="black", markerfacecolor="green", markersize=0),
            matl.Line2D([0], [0], linestyle="none", marker=".", color="blue", markerfacecolor="blue", markersize=9)
        ]
        matp.legend(legend, legend_description, fontsize="7", loc="upper left").get_frame().set_linewidth(0.3)


    def draw_points(self):
        matp.plot(self.X1[:, 0], self.X1[:, 1], marker="x", markersize=5, color="red", linestyle="none")
        matp.plot(self.X2[:, 0], self.X2[:, 1], marker="o", markersize=4, color="green", linestyle="none")
        matp.scatter(self.SVM.support_vector[:, 0], self.SVM.support_vector[:, 1], s=60, color="blue")


    def f(self, x, w, b, c=0):
        return (-w[0] * x - b + c) / w[1]


    def draw_margin_boundary(self, c, string):
        x = [
            self.x1_min,
            self.x1_max
        ]
        y = [
            self.f(self.x1_min, self.SVM.weight, self.SVM.intercept, c), 
            self.f(self.x1_max, self.SVM.weight, self.SVM.intercept, c)
        ]
        matp.plot(x, y, string)


    def plot(self):
        fig = matp.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set(xlim=(self.x1_min, self.x1_max), ylim=(self.x2_min, self.x2_max))
        self.setup_labels()
        self.setup_legend()
        self.draw_points()
        self.draw_margin_boundary(0, "k")
        self.draw_margin_boundary(1, "k--")
        self.draw_margin_boundary(-1, "k--")
        matp.show()


class Dataset:

    def __init__(self, seed=5, percent=0.7):
        self.seed = seed
        self.percent = percent
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None


    def linear_dataset(self):
        xy = []
        numpy.random.seed(self.seed)
        cov = numpy.array([[0.8, 0.6], [0.6, 0.8]])
        mean1 = numpy.array([0, 2])
        xy.append(numpy.random.multivariate_normal(mean1, cov, 100))
        xy.append(numpy.ones(len(xy[0])))
        mean2 = numpy.array([2, 0])
        xy.append(numpy.random.multivariate_normal(mean2, cov, 100))
        xy.append(numpy.ones(len(xy[2])) * -1)
        
        xy_test = []
        xy_train = []
        cutOff = int(len(xy[0])*self.percent);
        for e in xy:
            xy_train.append(e[:cutOff])
            xy_test.append(e[cutOff:])

        self.x_train = numpy.vstack((xy_train[0], xy_train[2]))
        self.y_train = numpy.hstack((xy_train[1], xy_train[3]))
        self.x_test = numpy.vstack((xy_test[0], xy_test[2]))
        self.y_test = numpy.hstack((xy_test[1], xy_test[3]))


class Supportvectormachine:

    def __init__(self):
        self.alphas = []
        self.support_vector = []
        self.sv_y = []
        self.intercept = 0
        self.weight = None


    def linear_kernel(self, x1, x2):
        return numpy.dot(x1, x2)


    def predict(self, X):
        return numpy.sign(self.project(X))


    def project(self, X):
        if self.weight is not None:
            return numpy.dot(X, self.weight) + self.intercept
        else:
            y_predict = numpy.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.alphas, self.sv_y, self.support_vector):
                    s += a * sv_y * self.linear_kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.intercept


    def fit(self, X, y):
        n_samples, n_features = X.shape

        K = numpy.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.linear_kernel(X[i], X[j])

        cvxopt.solvers.options["show_progress"] = False
        cvxopt.solvers.options["abstol"] = 1e-10
        cvxopt.solvers.options["reltol"] = 1e-10
        cvxopt.solvers.options["feastol"] = 1e-10
        alphas = numpy.ravel(cvxopt.solvers.qp(
            cvxopt.matrix(numpy.outer(y, y) * K),
            cvxopt.matrix(numpy.ones(n_samples) * -1),
            cvxopt.matrix(numpy.diag(numpy.ones(n_samples) * -1)),
            cvxopt.matrix(numpy.zeros(n_samples)),
            cvxopt.matrix(y, (1, n_samples)),
            cvxopt.matrix(0.0)
            )["x"]
        )

        sv = alphas > 1e-5
        ind = numpy.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.support_vector = X[sv]
        self.sv_y = y[sv]
        for n in range(len(self.alphas)):
            self.intercept += self.sv_y[n] - numpy.sum(self.alphas*self.sv_y*K[ind[n], sv])
        self.intercept = self.intercept/len(self.alphas)
        self.weight = numpy.zeros(n_features)
        for n in range(len(self.alphas)):
            self.weight += self.alphas[n] * self.sv_y[n] * self.support_vector[n]


def main():
    print("\n" + "="*70)
    print(" Support Vector Machine Algrithm")
    print("="*70 + "\n")
    DS = Dataset()
    DS.linear_dataset()
    x_train, y_train, x_test, y_test = [DS.x_train, DS.y_train, DS.x_test, DS.y_test]
    SVM = Supportvectormachine()
    SVM.fit(x_train, y_train)
    y_predict = SVM.predict(x_test)
    correct_predictions = numpy.sum(y_predict == y_test)
    
    print(" [+] Support Vectors: \n")
    t = prettytable.PrettyTable(["# Support Vectors", "# Train Examples"])
    t.add_row([len(SVM.support_vector), len(x_train)])
    print(t)
    
    print("\n [+] Predictions: \n")
    t = prettytable.PrettyTable(["# Correct Predictions", "# Test Examples"])
    t.add_row([correct_predictions, len(x_test)])
    print(t)    

    print("\n [+] Evaluation Metric Scores: \n")
    t = prettytable.PrettyTable(["Metric", "Score"])
    t.add_row(["Accuracy", sklearn.metrics.accuracy_score(y_test, y_predict)])
    t.add_row(["F1", sklearn.metrics.f1_score(y_test, y_predict)])
    t.add_row(["Precision", sklearn.metrics.precision_score(y_test, y_predict)])
    t.add_row(["Recall", sklearn.metrics.recall_score(y_test, y_predict)])
    print(t)    

    Plot(x_train[y_train == 1], x_train[y_train == -1], SVM).plot()


main()


```

### Output

    
    ======================================================================
     Support Vector Machine Algrithm
    ======================================================================
    
     [+] Support Vectors: 
    
    +-------------------+------------------+
    | # Support Vectors | # Train Examples |
    +-------------------+------------------+
    |         3         |       140        |
    +-------------------+------------------+
    
     [+] Predictions: 
    
    +-----------------------+-----------------+
    | # Correct Predictions | # Test Examples |
    +-----------------------+-----------------+
    |           60          |        60       |
    +-----------------------+-----------------+
    
     [+] Evaluation Metric Scores: 
    
    +-----------+-------+
    |   Metric  | Score |
    +-----------+-------+
    |  Accuracy |  1.0  |
    |     F1    |  1.0  |
    | Precision |  1.0  |
    |   Recall  |  1.0  |
    +-----------+-------+



![output_26_1](https://user-images.githubusercontent.com/86275885/124319783-c5f18480-db48-11eb-9a3a-49cdcadd8077.png)



