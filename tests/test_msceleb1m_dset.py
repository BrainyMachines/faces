import os
import unittest
import torchvision.datasets as datasets
import logging
from PIL import Image

class DatasetTester(unittest.TestCase):

    data_root = os.path.expanduser('/data/datasets')
    img_root = os.path.expanduser('/data/datasets/msceleb1m_images')

    def test_msceleb1m_dset(self):
        """Test Case for accessing MSCeleb1M dataset."""
        dset = datasets.MSCeleb1M(self.data_root, self.img_root, 'msceleb1m',
                                  download=False)
        img, target = dset[898]
        img.save('test_image.jpg')
        print('#images:', len(dset))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
