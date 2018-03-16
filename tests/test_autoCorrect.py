import autoCorrect
import numpy
import unittest

# Test whether the correction works.
class TestEndToEnd(unittest.TestCase):

    def test_end_to_end(self):
        counts = numpy.random.negative_binomial(n = 20, p=0.2, size = (10,8))
        sf = numpy.ones((10,1))
        corrector = autoCorrect.correctors.AECorrector()
        corrector.correct(counts = counts, size_factors = sf)
        self.assertEqual(counts.shape, correction.shape)

if __name__ == '__main__':
    unittest.main()
