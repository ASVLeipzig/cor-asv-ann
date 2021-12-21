# -*- coding: utf-8 -*-
"""
Installs:
    - cor-asv-ann-compare
    - cor-asv-ann-eval
    - cor-asv-ann-proc
    - cor-asv-ann-train
    - cor-asv-ann-repl
    - ocrd-cor-asv-ann-process
    - ocrd-cor-asv-ann-evaluate
    - ocrd-cor-asv-ann-align
"""
import codecs

from setuptools import setup, find_packages
import json

install_requires = open('requirements.txt').read().split('\n')

with codecs.open('README.md', encoding='utf-8') as f:
    README = f.read()

with open('./ocrd-tool.json', 'r') as f:
    version = json.load(f)['version']

setup(
    name='ocrd_cor_asv_ann',
    version=version,
    description='sequence-to-sequence translator for noisy channel error correction',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Robert Sachunsky',
    author_email='sachunsky@informatik.uni-leipzig.de',
    url='https://github.com/ASVLeipzig/cor-asv-ann',
    license='Apache License 2.0',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=install_requires,
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'cor-asv-ann-compare=ocrd_cor_asv_ann.scripts.compare:cli',
            'cor-asv-ann-train=ocrd_cor_asv_ann.scripts.train:cli',
            'cor-asv-ann-eval=ocrd_cor_asv_ann.scripts.eval:cli',
            'cor-asv-ann-proc=ocrd_cor_asv_ann.scripts.proc:cli',
            'cor-asv-ann-repl=ocrd_cor_asv_ann.scripts.repl:cli',
            'ocrd-cor-asv-ann-process=ocrd_cor_asv_ann.wrapper.transcode:ocrd_cor_asv_ann_process',
            'ocrd-cor-asv-ann-evaluate=ocrd_cor_asv_ann.wrapper.evaluate:ocrd_cor_asv_ann_evaluate',
            'ocrd-cor-asv-ann-align=ocrd_cor_asv_ann.wrapper.align:ocrd_cor_asv_ann_align',
        ]
    },
)
