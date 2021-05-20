import unittest
import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from hst.hst import HST


class ComputeSizes(unittest.TestCase):
    def test_piston_dia(self):
        self.assertEqual(round(test_hst.sizes['d'], ndigits=4), 0.0225)

    def test_pcd(self):
        self.assertEqual(round(test_hst.sizes['D'], ndigits=4), 0.0860)

    def test_shoe_outer_rad(self):
        self.assertEqual(round(test_hst.sizes['Rs'], ndigits=4), 0.0137)


class LoadOil(unittest.TestCase):
    def test_oil_data_shape(self):
        self.assertEqual(test_hst.oil_data.shape, (11, 3))

    def test_oil_data_columns(self):
        self.assertEqual(list(test_hst.oil_data.columns),
                         ['Dyn. Viscosity', 'Kin. Viscosity', 'Density'])


class ComputeEfficiency(unittest.TestCase):
    def test_motor_speed(self):
        self.assertEqual(round(test_performance['motor']['speed'], ndigits=3),
                         2068.024)

    def test_motor_torque(self):
        self.assertEqual(round(test_performance['motor']['torque'], ndigits=3),
                         489.130)

    def test_motor_power(self):
        self.assertEqual(round(test_performance['motor']['power'], ndigits=3),
                         105.927)

    def test_hst_volumetric(self):
        self.assertEqual(
            round(test_efficiency['hst']['volumetric'], ndigits=3), 68.934)

    def test_hst_mechanical(self):
        self.assertEqual(
            round(test_efficiency['hst']['mechanical'], ndigits=3), 92.266)

    def test_hst_total(self):
        self.assertEqual(round(test_efficiency['hst']['total'], ndigits=3),
                         63.603)


if __name__ == '__main__':
    test_hst = HST()
    test_hst.compute_sizes(100)
    test_hst.load_oil()
    test_efficiency, test_performance = test_hst.compute_eff(3000, 350)
    unittest.main(verbosity=2)