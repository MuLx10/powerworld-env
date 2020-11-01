from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='gym_powerworld',
  version='0.0.1',
  description=('OpenAI gym environment for interfacing with PowerWorld Simulator'),
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/mulx10/powerworld-env',
  author='Mehul Kumar Nirala',
  author_email='mehulkumarnirala@gmail.com',
  classifiers=[
    'Development Status :: 1 - Planning',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8'
  ],
  keywords=('deep reinforcement learning machine PowerWorld smart grid voltage control electric power system'),
  python_requires='>=3.5',
  install_requires=[
    'gym==0.15.4',
    'numpy==1.18.1',
    'pandas==1.0.1',
    'esa==0.6.2',
    'pillow==7.0.0',
    'matplotlib==3.1.2'
  ]
)