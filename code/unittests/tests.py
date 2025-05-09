import unittest
import sys
import os

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from acdc import check_acdc_epochs, get_acdc_phase, acdc_train_network


class TestACDC(unittest.TestCase):
    def test_check_acdc_phases(self):
        check_acdc_epochs(warm_up_epochs=10, alternating_epochs=20, convergence_epochs=30, total_epochs=200)

    def test_check_acdc_phases2(self):
        check_acdc_epochs(warm_up_epochs=10, alternating_epochs=3, convergence_epochs=8, total_epochs=30)

    def test_check_acdc_phases_incorrect(self):
        with self.assertRaises(ValueError):
            check_acdc_epochs(warm_up_epochs=30, alternating_epochs=3, convergence_epochs=8, total_epochs=30)

    def test_get_acdc_phase(self):
        self.assertEqual(get_acdc_phase(epoch=1, warm_up_epochs=10, alternating_epochs=20, convergence_epochs=30, total_epochs=200), 'dense')
        self.assertEqual(get_acdc_phase(epoch=10, warm_up_epochs=10, alternating_epochs=20, convergence_epochs=30, total_epochs=200), 'dense')
        self.assertEqual(get_acdc_phase(epoch=11, warm_up_epochs=10, alternating_epochs=20, convergence_epochs=30, total_epochs=200), 'sparse')
        self.assertEqual(get_acdc_phase(epoch=40, warm_up_epochs=10, alternating_epochs=20, convergence_epochs=30, total_epochs=200), 'dense')
        self.assertEqual(get_acdc_phase(epoch=70, warm_up_epochs=10, alternating_epochs=20, convergence_epochs=30, total_epochs=200), 'sparse')
        self.assertEqual(get_acdc_phase(epoch=170, warm_up_epochs=10, alternating_epochs=20, convergence_epochs=30, total_epochs=200), 'dense')
        self.assertEqual(get_acdc_phase(epoch=171, warm_up_epochs=10, alternating_epochs=20, convergence_epochs=30, total_epochs=200), 'sparse')
        self.assertEqual(get_acdc_phase(epoch=200, warm_up_epochs=10, alternating_epochs=20, convergence_epochs=30, total_epochs=200), 'sparse')

if __name__ == '__main__':
    unittest.main()
