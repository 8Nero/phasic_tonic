# Phasic-Tonic
[![PyPI version](https://badge.fury.io/py/phasic_tonic.svg)](https://badge.fury.io/py/phasic_tonic)
[![Build Status](https://travis-ci.org/yourusername/phasic_tonic.svg?branch=main)](https://travis-ci.org/yourusername/phasic_tonic)
[![License](https://img.shields.io/github/license/8Nero/phasic_tonic)](LICENSE)

## Overview

`phasic_tonic` is a python package for analysing phasic and tonic REM states from electrophysiological signals. 
It implements a threshold-based signal processing algorithm for detecting phasic substates within REM sleep.
`phasic_tonic` is intended for researchers working with sleep data and looking to classify REM sleep into more granular substates for further study.
This package was developed by [Tuguldur Gerelmaa](https://github.com/8Nero) and [Abdel Rayan](https://github.com/AbdelRayan/) for phasic and tonic research in Genzel lab.

<!-- ![pic1](detect_phasic_001.png) -->
<p align="center">
  <img width="70%" src="docs/images/detect_phasic_001.png">
</p>

<!-- ![pic1](detect_phasic_002.png) -->
<p align="center">
  <img width="70%" src="docs/images/detect_phasic_002.png">
</p>

## Key Features

- **Phasic/Tonic Detection**: Applies threshold-based algorithms to distinguish phasic and tonic states from raw electrophysiological data
- **Statistical Analysis**: Compute basic statistics for phasic/tonic REM periods.

## Installation

You can install **phasic_tonic** from PyPI using pip:
```bash
pip install phasic_tonic
```
or from the source code:
``` {.sourceCode .shell}
$ git clone https://github.com/8Nero/phasic_tonic.git
$ cd phasic_tonic
$ pip install -e .
```

## Package dependencies

+ numpy
+ scipy
+ neurodsp
+ pynapple

## Quick Start

Here's a simple example of how to get started with phasic_tonic:

```py
from phasic_tonic.detect import detect_phasic

phasicREM = detect_phasic(signal, hypnogram, fs)
```
