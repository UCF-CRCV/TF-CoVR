<h1 align="center"> From Play to Replay: Composed Video Retrieval for
Temporally Fine-Grained Videos</h1>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="tf-covr">
</p>

<p align="center">
  <a href="https://animesh-007.github.io/">Animesh Gupta</a> &nbsp;|&nbsp;
  <a href="https://www.linkedin.com/in/jay-himmatbhai-parmar/">Jay Parmar</a> &nbsp;|&nbsp;
  <a href="https://www.linkedin.com/in/ishan-dave-crcv/">Ishan Rajendrakumar Dave</a> &nbsp;|&nbsp;
  <a href="https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en&oi=ao">Mubarak Shah</a>
</p>


#### University of Central Florida

<p align="center">
  <a href="https://arxiv.org/abs/2506.05274">
    <img src="https://img.shields.io/badge/arXiv-TFCoVR-9065CA.svg?logo=arXiv" alt="arXiv">
  </a>
  <a href="https://animesh-007.github.io/TF-CoVR-WEBSITE/">
    <img src="https://img.shields.io/badge/Project-Page-orange?logo=data:image/svg%2bxml;base64,...(truncated)" alt="Project Page">
  </a>
  <a href="https://github.com/UCF-CRCV/TF-CoVR/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  </a>
</p>

<h5 align="left"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

#### Official GitHub repository for  `From Play to Replay: Composed Video Retrieval for Temporally Fine-Grained Videos`.

![tfcovr teaser gif](assets/teaser.gif)

> Composed Video Retrieval (CoVR) retrieves a target video given a query video and a modification text describing the intended change. Existing CoVR benchmarks emphasize appearance shifts or coarse event changes and therefore do not test the ability to capture subtle, fast-paced temporal differences. We introduce **TF-CoVR**, the first large-scale benchmark dedicated to temporally fine-grained CoVR. **TF-CoVR** focuses on gymnastics and diving and provides 180K triplets drawn from FineGym and FineDiving. Previous CoVR benchmarks focusing on temporal aspect, link each query to a single target segment taken from the same video, limiting practical usefulness. In **TF-CoVR**, we instead construct each <query, modification> pair by prompting an LLM with the label differences between clips drawn from different videos; every pair is thus associated with multiple valid target videos (3.9 on average), reflecting real-world tasks such as sports-highlight generation. To model these temporal dynamics we propose **TF-CoVR-Base**, a concise two-stage training framework: (i) pre-train a video encoder on fine-grained action classification to obtain temporally discriminative embeddings; (ii) align the composed query with candidate videos using contrastive learning. We conduct the first comprehensive study of image, video, and general multimodal embedding (GME) models on temporally fine-grained composed retrieval in both zero-shot and fine-tuning regimes. On TF-CoVR, TF-CoVR-Base improves zero-shot mAP@50 from 5.92 (LanguageBind) to 7.51, and after fine-tuning raises the state-of-the-art from 19.83 to 25.82.

## Environment Setup
```
cd TF-CoVR/
conda create -n tfcovr python=3.10 -y
conda activate tfcovr
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## Dataset
To be released.

## Training

### For reproducing results on TF-CoVR using TF-CoVR-Base

Run following command:  
`python train.py data=finegd-covr-aim trainer=gpu model=aim model/ckpt=aim test=finegd-test-aim`

## Testing  
`python test.py data=finegd-covr-aim trainer=gpu model=aim_clip model/ckpt=aim test=finegd-test-aim-clip machine.num_workers=8 trainer.max_epochs=100 model.ckpt.path=/checkpoint/path/`

## Citation
If you use this dataset and/or this code in your work, please cite our [paper](https://arxiv.org/abs/2506.05274):

```bibtex
@misc{gupta2025playreplaycomposedvideo,
      title={From Play to Replay: Composed Video Retrieval for Temporally Fine-Grained Videos}, 
      author={Animesh Gupta and Jay Parmar and Ishan Rajendrakumar Dave and Mubarak Shah},
      year={2025},
      eprint={2506.05274},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.05274}, 
}
```

## 🙏 Acknowledgements

This repository has borrowed code from [CoVR](https://github.com/lucas-ventura/CoVR). We thank the authors for releasing their code.

---

<p align="center">
   <a href="https://www.crcv.ucf.edu/"><img src="assets/crcv_ucf.jpg" width="500" height="90"></a>
</p>
