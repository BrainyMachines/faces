import os
import unittest
import torchvision.datasets as datasets
import logging
from PIL import Image

class DatasetTester(unittest.TestCase):

    root = os.path.expanduser('/data/datasets')

    def test_msceleb1m_dset(self):
        """Test Case for accessing MSCeleb1M dataset."""
        dset = datasets.MSCeleb1M(self.root, 'msceleb1m', download=False)
        img, target = dset[898]
        img.save('test_image.jpg')
        print('#images:', len(dset))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()