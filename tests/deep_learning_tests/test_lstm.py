import unittest
import numpy as np
from deep_learning.lstm import MyLSTM


class TestMyLSTM(unittest.TestCase):
    def setUp(self):
        self.sequence_length = 10
        self.input_dim = 5
        self.dataset_name = "test_dataset"
        self.model = MyLSTM(
            sequence_length=self.sequence_length,
            input_dim=self.input_dim,
            dataset_name=self.dataset_name,
            cells=[50, 20],
            num_classes=2,
            multi_class=False,
            batch_size=32,
            epochs=10
        )
        self.model.build()

    def test_model_name(self):
        name = self.model.model_name()
        self.assertIn("lstm", name)
        self.assertIn("sl-10", name)
        self.assertIn("bc", name)
        self.assertIn("layers-50-20", name)

    def test_build(self):
        self.assertIsNotNone(self.model.model)
        self.assertEqual(len(self.model.model.layers), 4)
        
    def test_create_sequences(self):
        data = np.random.rand(100, self.input_dim)
        labels = np.random.randint(2, size=100)

        # Test for sequence_length = 1
        self.model.sequence_length = 1
        x_seq, y_seq = self.model.create_sequences(data, labels)
        self.assertEqual(x_seq.shape[0], data.shape[0])
        self.assertEqual(x_seq.shape[1], 1)
        self.assertEqual(x_seq.shape[2], self.input_dim)
        self.assertEqual(y_seq.shape[0], labels.shape[0])

        # Test for sequence_length > 1
        self.model.sequence_length = 10
        x_seq, y_seq = self.model.create_sequences(data, labels)
        self.assertEqual(x_seq.shape[0], 91)  # 100 - 10 + 1
        self.assertEqual(x_seq.shape[1], 10)
        self.assertEqual(x_seq.shape[2], self.input_dim)
        self.assertEqual(y_seq.shape[0], 91)

    def test_train_and_predict(self):
        data = np.random.rand(100, self.input_dim)
        labels = np.random.randint(2, size=100)

        self.model.train(data, labels)
        predictions, time_taken = self.model.predict(data)

        self.assertEqual(len(predictions), 91)  # 100 - 10 + 1
        self.assertGreater(time_taken, 0)

    def test_evaluate(self):
        predictions = np.array([0, 1, 0, 1])
        labels = np.array([0, 1, 0, 1])
        _, scores, _ = self.model.evaluate(predictions, labels, 0.1, verbose=0)

        self.assertAlmostEqual(scores["accuracy"], 1.0)
        self.assertAlmostEqual(scores["recall"], 1.0)
        self.assertAlmostEqual(scores["precision"], 1.0)
        self.assertAlmostEqual(scores["f1s"], 1.0)


if __name__ == '__main__':
    unittest.main()
