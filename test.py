import unittest
from model import ModelPipeline

class TestModelPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = ModelPipeline()

    def test_prediction_output_type(self):
        class_id = self.pipeline.predict_from_path('n01667114_mud_turtle.JPEG')
        self.assertIsInstance(class_id, int)

    def test_prediction_output_value_range(self):
        class_id = self.pipeline.predict_from_path('n01667114_mud_turtle.JPEG')
        self.assertTrue(0 <= class_id < 1000)

    def test_correct_class(self):
        # This image is known to belong to class 35
        class_id = self.pipeline.predict_from_path('n01667114_mud_turtle.JPEG')
        self.assertEqual(class_id, 35)
        
        
        
    def test_prediction_output_type(self):
        class_id = self.pipeline.predict_from_path('n01440764_tench.jpeg')
        self.assertIsInstance(class_id, int)

    def test_prediction_output_value_range(self):
        class_id = self.pipeline.predict_from_path('n01440764_tench.jpeg')
        self.assertTrue(0 <= class_id < 1000)

    def test_correct_class(self):
        # This image is known to belong to class 35
        class_id = self.pipeline.predict_from_path('n01440764_tench.jpeg')
        self.assertEqual(class_id, 0)

if __name__ == "__main__":
    unittest.main()
