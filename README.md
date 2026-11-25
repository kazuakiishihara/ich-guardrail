# ICH Guardrail

Code for automated hematoma segmentation and LLM input generation from non-contrast head CT.

## Dataset

The hematoma segmentation model is trained on **HemSeg-200** ([Song et al., 2024](https://arxiv.org/abs/2405.14559)), a dataset of voxel-level annotations for brain CT scans selected from the **RSNA 2019 Brain CT Hemorrhage Challenge** dataset ([Flanders et al., 2020](https://doi.org/10.1148/ryai.2020190211)).

```bibtex
@article{song2024hemseg,
  title={HemSeg-200: A Voxel-Annotated Dataset for Intracerebral Hemorrhages Segmentation in Brain CT Scans},
  author={Song, Changwei and Zhao, Qing and Li, Jianqiang and Yue, Xin and Gao, Ruoyun and Wang, Zhaoxuan and Gao, An and Fu, Guanghui},
  journal={arXiv preprint arXiv:2405.14559},
  year={2024}
}
```

```bibtex
@article{flanders2020construction,
  title={Construction of a machine learning dataset through collaboration: the RSNA 2019 brain CT hemorrhage challenge},
  author={Flanders, Adam E and Prevedello, Luciano M and Shih, George and Halabi, Safwan S and Kalpathy-Cramer, Jayashree and Ball, Robyn and Mongan, John T and Stein, Anouk and Kitamura, Felipe C and Lungren, Matthew P and others},
  journal={Radiology: Artificial Intelligence},
  volume={2},
  number={3},
  pages={e190211},
  year={2020},
  publisher={Radiological Society of North America}
}
```
