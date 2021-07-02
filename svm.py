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

