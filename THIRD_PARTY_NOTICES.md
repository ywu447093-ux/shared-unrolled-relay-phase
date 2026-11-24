# Third-party research sources

## SiSPRNet

The original author model is **not redistributed** in this repository. Its
upstream repository does not display a general redistribution license.
Obtain the source from its authors under their applicable terms before using
the SiSPRNet adaptation. The retained comparison used:

- https://github.com/Qiustander/SiPRNet/blob/main/Model/model_SiPRNet.py
- Required local destination: `work/experiments/vendor/model_SiPRNet_original.py`
- SHA-256 of the file used in the research: `4006a071855905b32d8a0aa702b02ceb937616b940ededea5aa4a7b41b7790c8`
- Q.Ye, L.-W.Wang and D.P.K.Lun, Optics Express30,31937–31958(2022), doi:10.1364/OE.464086.

## FourierGSNet

`work/reviewer_validation_v1/fouriergs_vendor/model_GSNet.py` is distributed
with the upstream MIT license at the same path. Copyright2025 Shengyuan Yan.
Upstream: https://github.com/yanshengyuan/FourierGSNet,
commit `e4f74bc720ab344d2f427f7f27968c5eff4eca0b`.
The source-known conditional-network variant is adapted to this acquisition;
it is not the original paper's reported experiment.

## PhysenNet

`work/experiments/physennet.py` is this study's PyTorch implementation of the
published U-Net architecture and per-observation fitting adaptation. The
upstream TensorFlow file is not redistributed here. Upstream source:
https://github.com/FeiWang0824/PhysenNet (academic/non-commercial terms).
Reference: F.Wang et al., Light:Science & Applications9,77(2020),
doi:10.1038/s41377-020-0302-3.

## Publication and licensing

This release makes the research source publicly inspectable. No new blanket
license has been selected for project-authored code; the upstream MIT license
applies only to its identified third-party files. No additional rights to
other authors' source or data are asserted by this repository.
