import unittest
import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

import hst_analysis

loaded_models, data = hst_analysis.load_catalogues()
fitted_models = hst_analysis.fit_catalogues(data)


class TestCatalogueModels(unittest.TestCase):
    def test_loaded_models_number(self):
        self.assertEqual(len(loaded_models),
                         4,
                         msg='Models number is incorrect.')

    def test_data_shape(self):
        self.assertEqual(data.shape, (207, 5), msg='Data shape is incorrect.')

    def test_fitted_models_number(self):
        self.assertEqual(len(fitted_models),
                         4,
                         msg='Models number is incorrect.')


if __name__ == '__main__':
    unittest.main(verbosity=2)