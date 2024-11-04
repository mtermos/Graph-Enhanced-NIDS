import unittest
import numpy as np
from src.deep_learning.model import Model


class DummyModel(Model):
    def model_name(self):
        return "dummy_model"

    def build(self):
        pass

    def train(self):
        pass

    def predict(self):
        return [0, 1, 0, 1], 0.1


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = DummyModel(sequential=True)

    def test_evaluate_binary(self):
        predictions = np.array([0, 1, 0, 1])
        labels = np.array([0, 1, 0, 1])
        _, scores, _ = self.model.evaluate(predictions, labels, 0.1, verbose=0)

        self.assertAlmostEqual(scores["accuracy"], 1.0)
        self.assertAlmostEqual(scores["recall"], 1.0)
        self.assertAlmostEqual(scores["precision"], 1.0)
        self.assertAlmostEqual(scores["f1s"], 1.0)
        self.assertAlmostEqual(scores["FPR"], 0.0)
        self.assertAlmostEqual(scores["FNR"], 0.0)

    def test_evaluate_multi_class(self):
        self.model.multi_class = True
        predictions = np.array([0, 1, 2, 1])
        labels = np.array([0, 1, 2, 1])
        _, scores, _ = self.model.evaluate(predictions, labels, 0.1, verbose=0)

        self.assertAlmostEqual(scores["accuracy"], 1.0)
        self.assertAlmostEqual(scores["recall"], 1.0)
        self.assertAlmostEqual(scores["precision"], 1.0)
        self.assertAlmostEqual(scores["f1s"], 1.0)
        self.assertAlmostEqual(scores["FPR"], 0.0)
        self.assertAlmostEqual(scores["FNR"], 0.0)


if __name__ == '__main__':
    unittest.main()
