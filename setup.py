from setuptools import setup

setup(name='riples-classifier',
      version='0.1',
      description='Unified classifier framework for RiPLes project code.',
      packages=['riples_classifier'],
      install_requires=[
            'scikit-learn>=0.18',
            'scipy>=1.0',
      ])