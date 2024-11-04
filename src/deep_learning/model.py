import os
import json
import numpy as np
from abc import ABCMeta, abstractmethod
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, classification_report


class Model(metaclass=ABCMeta):

    def __init__(self, sequential=False, multi_class=False):
        self.sequential = sequential
        self.multi_class = multi_class

    @abstractmethod
    def model_name(self): raise NotImplementedError

    @abstractmethod
    def build(self): raise NotImplementedError

    @abstractmethod
    def train(self): raise NotImplementedError

    @abstractmethod
    def predict(self): raise NotImplementedError

    def evaluate(self, predictions, labels, time, verbose=0):
        model_name = self.model_name()

        average = "weighted"

        accuracy = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions, average=average)
        precision = precision_score(labels, predictions, average=average)
        f1s = f1_score(labels, predictions, average=average)
        cm = confusion_matrix(labels, predictions)
        class_report = classification_report(labels, predictions)

        if verbose == 1:
            print("model: ", model_name)

            print("Confusion Matrix:")
            print(cm)
            print("End of Confusion Matrix:")

            print("Classification Report:")
            print(class_report)
            print("End of Classification Report:")

        if self.multi_class:
            TN, FP, FN, TP, class_report = self._multi_class_metrics(cm)

        else:
            TN, FP, FN, TP = self._binary_class_metrics(cm)

            if cm.shape[0] == 1 and cm.shape[1] == 1:
                return (model_name, {
                    "accuracy": accuracy,
                    "recall": recall,
                    "precision": precision,
                    "f1s": f1s,
                    "FPR": 0,
                    "FNR": 0,
                    "time": time
                })

        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP)
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)

        if verbose == 1:
            print("Scores:")
            print("Accuracy: " + "{:.3%}".format(accuracy))
            print("Recall: " + "{:.3%}".format(recall))
            print("Precision: " + "{:.3%}".format(precision))
            print("F1-Score: " + "{:.3%}".format(f1s))
            print("True positive: " + "{}".format(TP))
            print("True negative: " + "{}".format(TN))
            print("False positive: " + "{}".format(FP))
            print("False negative: " + "{}".format(FN))
            print("False positive rate: " + "{:.3%}".format(FPR))
            print("False negative rate: " + "{:.3%}".format(FNR))
            print("Prediction time: " + "{:.3%}".format(time))
            print("End of Scores:")
            print("======================================")

        scores = {
            "accuracy": accuracy,
            "recall": recall,
            "precision": precision,
            "f1s": f1s,
            "FPR": FPR,
            "FNR": FNR,
            "time": time
        }
        return (model_name, scores, class_report)

    def _multi_class_metrics(self, cm):
        # Multi-class specific metrics
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)

        FP = FP.astype(float).sum()
        FN = FN.astype(float).sum()
        TP = TP.astype(float).sum()
        TN = TN.astype(float).sum()

        records_count = {}
        class_accuracy = {}
        class_precision = {}
        class_recall = {}
        class_f1 = {}
        class_fnr = {}
        class_fpr = {}
        for i in range(cm.shape[0]):
            # True positive, true negative, false positive, false negative
            tp = cm[i, i]
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp

            # Num of records in this class
            records_count[i] = tp + fn

            # Accuracy for the class
            records_count[i] = tp + fn
            class_accuracy[i] = (tp + tn) / (tp + tn + fp + fn)
            class_precision[i] = tp / (tp + fp)
            class_recall[i] = tp / (tp + fn)
            class_f1[i] = 2 * (class_precision[i] * class_recall[i]) / \
                (class_precision[i] + class_recall[i])
            class_fnr[i] = fn / (fn + tp)
            class_fpr[i] = fp / (fp + tn)

        class_report = {
            "records_count": records_count,
            "class_accuracy": class_accuracy,
            "class_precision": class_precision,
            "class_recall": class_recall,
            "class_f1": class_f1,
            "class_fnr": class_fnr,
            "class_fpr": class_fpr
        }
        return TN, FP, FN, TP, class_report

    def _binary_class_metrics(self, cm):
        # TN, FP, FN, TP = cm.ravel()

        TN = cm[0][0]
        FN = cm[1][0]
        TP = cm[1][1]
        FP = cm[0][1]

        return TN, FP, FN, TP

    def makedirs_and_data(self, weights_folder, logs_folder):
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    # Try to serialize the object using the default method
                    return super().default(obj)
                except TypeError:
                    # Check if the object has a to_json method and try to use it
                    if hasattr(obj, 'to_json') and callable(getattr(obj, 'to_json')):
                        return json.loads(obj.to_json())
                    # Fallback to string representation
                    return str(obj)

        os.makedirs(weights_folder, exist_ok=True)
        os.makedirs(logs_folder, exist_ok=True)

        with open(f"{weights_folder}/model.json", 'w') as f:
            json.dump(self.__dict__, f, cls=CustomEncoder)

        with open(f"{logs_folder}/model.json", 'w') as f:
            json.dump(self.__dict__, f, cls=CustomEncoder)
