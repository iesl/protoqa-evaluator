from distutils.core import setup
setup(name='family_feud_evaluator',
      version='0.2',
      py_modules=['family_feud_evaluator'],
      install_requires=[
            'scipy', 'numpy', 'nltk',
      ],
      extras_require={
            'test': ['pytest'],
            'crowdsource_conversion': ['pandas', 'xlrd'],
      }
      )