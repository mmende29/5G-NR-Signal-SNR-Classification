# 5G-NR Signal SNR Classification Using Deep Learning

A PyTorch deep learning pipeline designed to classify Signal-to-Noise Ratio (SNR) levels from 2-channel In-Phase and Quadrature (I/Q) 5G-NR RF signal sequences. 

## Overview

Signal-to-Noise Ratio (SNR) estimation is a critical component of modern wireless communications, link adaptation, and spectrum monitoring. This project processes raw RF binary data, extracts complex I/Q streams, normalizes signal parameters, injects Additive White Gaussian Noise (AWGN) to emulate varying real-world SNR environments, and classifies the signal conditions using a custom 1D Convolutional Neural Network (CNN).

This repository demonstrates the intersection of **Radio Frequency (RF) principles, digital signal processing, and machine learning**.

## Pipeline Architecture

1. **RF Processing & Parsing:** Extracts 16-bit binary I/Q channels captured at a 20 MHz sampling rate ($628\text{ MHz}$ center frequency).
2. **Phase-Preserving Normalization:** Scales I/Q signals by peak magnitude while rigorously preserving the original phase relationship.
3. **Noise Synthesis:** Computes base SNR and applies targeted Gaussian noise scaling across 10,000 sequence samples ($2000$ sequence length per channel).
4. **1D CNN Classification:** Evaluates dynamic temporal feature maps through stacked `Conv1D`, `MaxPool1d`, and `Dropout` layers, utilizing `skorch` for the training loop.

## Dataset Parameters

* **Source Data:** IEEE Dataport 5G-NR RF Dataset
* **Sampling Rate:** 20 MS/s
* **Bit Depth:** 16-bit integer
* **Input Dimension:** (10000, 2, 2000) -> [Batch Size, Channels, Sequence Length]
* **Target Classes:** High (20 dB), Medium (10 dB), Low (0 dB)

## Repository Structure

* `/data/`: Contains the raw IEEE Dataport RF data (ignored via `.gitignore` for size).
* `/src/`: Contains the modular PyTorch CNN architecture and signal processing utility functions.
* `/notebooks/`: Contains the primary execution pipeline.

## Development Environment

This project was developed and evaluated using **Spyder IDE** with its notebook environment integration. 

### Quickstart

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/5G-NR-Signal-SNR-Classification.git](https://github.com/your-username/5G-NR-Signal-SNR-Classification.git)
cd 5G-NR-Signal-SNR-Classification
pip install -r requirements.txt
