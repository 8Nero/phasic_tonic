---
hide:
  - toc
---
# Phasic-Tonic
[![PyPI version](https://badge.fury.io/py/phasic_tonic.svg)](https://badge.fury.io/py/phasic_tonic)
[![Build Status](https://travis-ci.org/yourusername/phasic_tonic.svg?branch=main)](https://travis-ci.org/yourusername/phasic_tonic)
[![License](https://img.shields.io/github/license/8Nero/phasic_tonic)](LICENSE)

## Overview

`phasic_tonic` is a python package for analysing phasic and tonic REM states from electrophysiological signals.
It implements a threshold-based signal processing algorithm for detecting phasic substates within REM sleep.

`phasic_tonic` is intended for researchers working with sleep data and looking to classify REM sleep into further substates.

## Examples

# <div style="text-align: center;"> <img src="images/detect_phasic_001.png" width="70%" alt="phasic tonic image."> </div>

# <div style="text-align: center;"> <img src="images/phasic_tonic_visualize.png" width="70%" alt="phasic tonic image."> </div>

## Key Features

- **Phasic/Tonic Detection**: Applies threshold-based algorithms to distinguish phasic and tonic states from raw electrophysiological data
- **Statistical Analysis**: Compute basic statistics for phasic/tonic REM periods.

## Quick Start

```py
import pooch
import numpy as np
from phasic_tonic.detect import detect_phasic

f_path = pooch.retrieve("https://raw.githubusercontent.com/8Nero/phasic_tonic/main/data/ex01.npz", 
                        known_hash="11e579b9305859db9101ba3e227e2d9a72008423b3ffe2ad9a8fee1765236663")

data = np.load(f_path, allow_pickle=True)

hypnogram = data['hypnogram']
lfp = data['lfp_hpc']
fs = 500  # Sampling rate

phasicREM = detect_phasic(lfp, hypnogram, fs)
print(phasicREM)
```