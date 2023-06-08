# BTS-E: Audio Deepfake Detection Using Breathing-Talking-Silence Encoder

## Code structure:
- biosegment: The Sound Segmentation Model, which uses 3 simple GMM for each class: Brathing, Talking and Silence.

- asvspoof2021/LA/Baseline-Rawnet2-bio: The full pipeline of our work.


# Reference
Specially thanks to [Hemlata Tak](https://scholar.google.co.in/citations?user=u2DMQxsAAAAJ&hl=en) and the ASVspoof organizers for publishing the baseline source code which is easy to follow. We use the Rawnet2 Baseline `https://github.com/asvspoof-challenge/2021/tree/main/LA/Baseline-RawNet2` for developing my system.

# How to cite
```
@INPROCEEDINGS{10095927,
  author={Doan, Thien-Phuc and Nguyen-Vu, Long and Jung, Souhwan and Hong, Kihun},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={BTS-E: Audio Deepfake Detection Using Breathing-Talking-Silence Encoder}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49357.2023.10095927}}
```