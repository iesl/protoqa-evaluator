"""
Evaluation framework for ProtoQA common sense QA dataset
"""
#    Copyright 2022 The ProtoQA Evaluator Authors.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import fastentrypoints
from setuptools import find_packages, setup

setup(
    name="protoqa_evaluator",
    version="1.1",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    package_dir={"": "src"},
    description="Evaluation scripts for ProtoQA common sense QA dataset.",
    install_requires=[
        "Click>=7.1.2",
        "scipy",
        "numpy",
        "nltk",
        "more-itertools",
        "xopen",
    ],
    extras_require={
        "test": ["pytest"],
        "crowdsource-conversion": ["pandas", "openpyxl"],
        "mlm-similarity": ["torch", "transformers", "scikit-learn"],
    },
    entry_points={
        "console_scripts": ["protoqa_evaluator = protoqa_evaluator.__main__:main"]
    },
)
