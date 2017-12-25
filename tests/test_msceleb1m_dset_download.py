import os
import tempfile
import unittest
import torchvision.datasets as datasets

class DatasetTester(unittest.TestCase):

    def test_msceleb1m_dset_nodownload(self):
        """Test Case: no download. download=False"""
        with tempfile.TemporaryDirectory(prefix='pytorch_') as root:
            with self.assertRaises(RuntimeError):
                datasets.MSCeleb1M(root, 'msceleb1m', download=False)

    def test_msceleb1m_dset_partialdownload(self):
        """Test Case: partial download. download=False"""
        num_bytes = 100
        with tempfile.TemporaryDirectory(prefix='pytorch_') as root:
            with tempfile.TemporaryFile(dir=root) as pfile:
                pfile.write(os.urandom(num_bytes))
                with self.assertRaises(RuntimeError):
                    datasets.MSCeleb1M(root, 'msceleb1m', download=False)

    def test_msceleb1m_dset_nodownload_dwld(self):
        """Test Case: no download. download=True"""
        with tempfile.TemporaryDirectory(prefix='pytorch_') as root:
            datasets.MSCeleb1M(root, 'msceleb1m', download=True)

    def test_msceleb1m_dset_partialdownload_dwld(self):
        """Test Case: partial download. download=True"""
        num_bytes = 100
        with tempfile.TemporaryDirectory(prefix='pytorch_') as root:
            with tempfile.TemporaryFile(dir=root) as pfile:
                pfile.write(os.urandom(num_bytes))
                datasets.MSCeleb1M(root, 'msceleb1m', download=True)


if __name__ == '__main__':
    unittest.main()