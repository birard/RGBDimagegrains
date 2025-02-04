# 📌 RGBDimagegrains

[![Issues](https://img.shields.io/github/issues/birard/RGBDimagegrains)](https://github.com/birard/RGBDimagegrains/issues)
[![Code Size](https://img.shields.io/github/languages/code-size/birard/RGBDimagegrains)](https://github.com/birard/RGBDimagegrains)


## 📖 Introduction

RGBDimagegrains is a grain size image recognition software that integrates RGB-D images, PebblecountAuto, and Imagegrains. It primarily utilizes RGB-D images to detect large grain sizes in advance and replaces labels in the deep learning model. The goal is to enhance segmentation in images of non-uniform gravel riverbeds.

## 🚀 Citation
 If you use software and/or data from here in your research, please cite the following works:
- Mair, D., Witz, G., Do Prado, A.H., Garefalakis, P. & Schlunegger, F. (2023) Automated detecting, segmenting and measuring of grains in images of fluvial sediments: The potential for large and precise data from specialist deep learning models and transfer learning. Earth Surface Processes and Landforms, 1–18. https://doi.org/10.1002/esp.5755.
- Stringer, C.A., Pachitariu, M., (2021). Cellpose: a generalist algorithm for cellular segmentation. Nat Methods 18, 100–106. https://doi.org/10.1038/s41592-020-01018-x.
- Benjamin, Purinton., Bodo, Bookhagen., (2019). Introducing PebbleCounts: a grain-sizing tool for photo surveys of dynamic gravel-bed rivers. Earth Surface Dynamics 7, 859-877. https://doi.org/10.1002/esp.5782.

### Prerequisites
- **Required software** ( Python )
### Prerequisites
- **Required software** (Python)

### Installation
```bash
conda create -n RGBDimagegrains python==3.8.17
conda activate RGBDimagegrains
git clone https://github.com/birard/RGBDimagegrains
cd your-dir
pip install -r requirements.txt
```
##### Usage
The following is a practical example of the operation process for RGBDimagegrains and RGBDgrains: https://www.youtube.com/watch?v=i9PZDbwDekc.
