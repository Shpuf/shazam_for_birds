import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

LABELS = np.load('labels.npy')

INPUTS = np.load('inputs.npy')


def main():
    clf = svm.SVC(gamma=0.00001)

    X_train, X_test, Y_train, Y_test = train_test_split(INPUTS, LABELS, test_size=0.33, shuffle=True)

    clf.fit(X_train, Y_train)

    predicted = clf.predict(X_test)

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(Y_test, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix of SVC")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()

    clf = MLPClassifier(solver='lbfgs', alpha=1e-4,
                        hidden_layer_sizes=(16, 8), random_state=1, max_iter=10000000)

    clf.fit(normalize(X_train), Y_train)

    predicted = clf.predict(normalize(X_test))

    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(Y_test, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(Y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix of NN")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()


if __name__ == '__main__':
    main()
