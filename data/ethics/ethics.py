# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data from the paper "Aligning AI With Shared Human Values, https://arxiv.org/abs/2008.02275"""


import csv
import json
import os

import datasets


_CITATION = """
@article{hendrycks2020aligning,
  title={Aligning ai with shared human values},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Critch, Andrew and Li, Jerry and Song, Dawn and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2008.02275},
  year={2020}
}
"""

_DESCRIPTION = """\
A benchmark that spans concepts in justice, well-being, duties, virtues, and commonsense morality.
"""

_HOMEPAGE = "https://github.com/hendrycks/ethics"

_LICENSE = "MIT"

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)

_URL_BASE = "https://huggingface.co/datasets/hendrycks/ethics/resolve/main/data/"
# _URL_SECTIONS = ["commonsense", "deontology", "justice", "utilitarianism", "virtue"]
_URL_ENDINGS = {
    "train": "train.csv",
    "test": "test.csv",
    "test_hard": "test_hard.csv",
}


class Ethics(datasets.GeneratorBasedBuilder):
    """A simple benchmark for aligning AI language systems."""

    VERSION = datasets.Version("1.0.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="commonsense", version=VERSION, description="This part of my dataset covers a first domain"),
        datasets.BuilderConfig(name="deontology", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="justice", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="utilitarianism", version=VERSION, description="This part of my dataset covers a second domain"),
        datasets.BuilderConfig(name="virtue", version=VERSION, description="This part of my dataset covers a second domain"),
    ]

    DEFAULT_CONFIG_NAME = "commonsense"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        section = self.config.name
        if section == "commonsense":  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "label": datasets.Value("int32"),
                    "input": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        elif section == "deontology":  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "label": datasets.Value("int32"),
                    "scenario": datasets.Value("string"),
                    "excuse": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        elif section == "justice":  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "label": datasets.Value("int32"),
                    "scenario": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        elif section == "utilitarianism":  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "baseline": datasets.Value("string"),
                    "less_pleasant": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        elif section == "virtue":  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "label": datasets.Value("int32"),
                    "scenario": datasets.Value("string"),
                    # These are the features of your dataset like images, labels ...
                }
            )
        else:
            raise ValueError(f"Data section {section} not in dataset")
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        base_url = _URL_BASE + self.config.name + "/"
        urls = { k:base_url + v for (k,v) in _URL_ENDINGS.items()}
        downloaded_files = dl_manager.download_and_extract(urls)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": downloaded_files["train"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": downloaded_files["test"]}
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": downloaded_files["test_hard"]}
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        # with open(filepath, encoding="utf-8") as f:
        with open(filepath, "r") as file:
            f = csv.reader(file)
            next(f)  # skips header
            for key, row in enumerate(f):   
                if self.config.name == "commonsense":
                    # Yields examples as (key, example) tuples
                    yield key, {
                        "input": row[1],
                        "label": row[0], 
                    }
                elif self.config.name == "deontology":
                    yield key, {
                        "scenario": row[1], 
                        "label": row[0], 
                        "excuse": row[2], 
                    }
                elif self.config.name == "justice":
                    yield key, {
                        "scenario": row[1], 
                        "label": row[0],
                    }
                elif self.config.name == "utilitarianism":
                    yield key, {
                        "baseline": row[0], 
                        "less_pleasant": row[1], 
                    }
                elif self.config.name == "virtue":
                    yield key, {
                        "scenario": row[1], 
                        "label": row[0], 
                    }
                else:
                    raise ValueError(f"Config name failed generating examples (not found).")