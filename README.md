# 🐋 Raven to SBE Info File Converter – Whale Localization Preprocessing Tool

This repository contains preprocessing code developed for a whale conservation research project in collaboration with **Dr. John Spiesberger** at the University of Pennsylvania. The scripts convert acoustic data from Raven output into a format suitable for SBE (Source Bearing Estimation) localization analysis.

## 🔍 Project Overview

Increased underwater noise from offshore wind energy and naval sonar is disrupting whale communication and migration. This project supports conservation efforts by enabling more accurate localization of whale vocalizations through passive acoustic monitoring.

## 🤝 Collaboration

This project was conducted under the guidance of **Dr. John Spiesberger**, whose research focuses on underwater acoustics and marine mammal localization. I worked closely with Dr. Spiesberger to develop tools that help analyze time-synchronized recordings collected from multiple ocean-bottom receivers.

## 🛠 What the Code Does

- Converts raw `.txt` files from the **Raven Pro** acoustic software into standardized info files for use with **SBE localization tools**.
- Parses and formats timestamped vocalization events across multiple hydrophones.
- Prepares data for time-delay-of-arrival (TDOA) analysis used in locating whale positions.

## 🧪 Tech Stack

- Python 3
- Built-in libraries (`os`, `csv`, `datetime`, etc.)

## 📁 File Structure

- `converter.py` – Main script to process Raven files.
- `example_data/` – Sample `.txt` input files.
- `output/` – Where formatted `.info` files are stored.

## 🌊 Research Context

This preprocessing pipeline was part of an ongoing oceanographic study that aims to track whale movement patterns in the presence of human-made noise. It contributes to sustainable offshore energy development and marine species protection.

## 👋 About Me

I’m **Yash Samat**, an AI student at the University of Pennsylvania passionate about combining tech with environmental sustainability. This project reflects my ongoing work in conservation-driven machine learning and acoustic signal processing.

## 📫 Contact

Feel free to reach out:
- Email: ysamat@seas.upenn.edu
- LinkedIn: [linkedin.com/in/ysamat](https://www.linkedin.com/in/ysamat)
