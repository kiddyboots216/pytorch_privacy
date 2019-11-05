from setuptools import find_packages, setup

setup(name='torchprivacy',
      version='0.0.1',
      url='https://github.com/kiddyboots216/pytorch_privacy',
      license='Apache-2.0',
      install_requires=['numpy>=1.16',
                #'mpmath>=1.1.0',
                'torch>=1.2.0',
                ],
      packages=find_packages(exclude=['docs']),
      )
