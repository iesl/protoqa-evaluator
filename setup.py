from distutils.core import setup
setup(name='family_feud_evaluator',
      version='0.2',
      py_modules=['family_feud_evaluator'],
      install_requires=[
            'scipy', 'numpy',
      ],
      extras_require={
            'test': ['pytest', 'nltk'],
            'crowdsource_conversion': ['pandas', 'xlrd'],
      }
      )