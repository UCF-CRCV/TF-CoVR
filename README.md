<h1 align="center"> From Play to Replay: Composed Video Retrieval for
Temporally Fine-Grained Videos</h1>

<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="tf-covr">
</p>

<p align="center">
  <a href="https://animesh-007.github.io/">Animesh Gupta<sup>1</sup></a> &nbsp;|&nbsp;
  <a href="https://www.linkedin.com/in/jay-himmatbhai-parmar/">Jay Parmar<sup>1</sup></a> &nbsp;|&nbsp;
  <a href="https://www.linkedin.com/in/ishan-dave-crcv/">Ishan Rajendrakumar Dave<sup>2</sup></a> &nbsp;|&nbsp;
  <a href="https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en&oi=ao">Mubarak Shah<sup>1</sup></a> <br><br>
<sup>1</sup>University of Central Florida&emsp;  <sup>2</sup>Adobe&emsp;
</p>

<div align="center">

[![](https://img.shields.io/badge/Project%20Page-ab99d7)](https://animesh-007.github.io/TF-CoVR-WEBSITE/)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2506.05274-b31b1b.svg)](https://arxiv.org/abs/2506.05274)&nbsp;
[![🤗 Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-TF--CoVR-orange)](https://huggingface.co/datasets/ucf-crcv/TF-CoVR)
[![License](https://img.shields.io/badge/License-MIT-green)](https://github.com/UCF-CRCV/TF-CoVR/blob/main/LICENSE)&nbsp;
![visitors](https://visitor-badge.laobi.icu/badge?page_id=UCF-CRCV/TF-CoVR)

---

</div>


<p align="center">
  <strong><em>Accepted in NeurIPS 2025</em></strong>
</p>


<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>


![tfcovr teaser gif](assets/teaser.gif)

> Composed Video Retrieval (CoVR) retrieves a target video given a query video and a modification text describing the intended change. Existing CoVR benchmarks emphasize appearance shifts or coarse event changes and therefore do not test the ability to capture subtle, fast-paced temporal differences. We introduce **TF-CoVR**, the first large-scale benchmark dedicated to temporally fine-grained CoVR. **TF-CoVR** focuses on gymnastics and diving and provides 180K triplets drawn from FineGym and FineDiving. Previous CoVR benchmarks focusing on temporal aspect, link each query to a single target segment taken from the same video, limiting practical usefulness. In TF-CoVR, we instead construct each <query, modification> pair by prompting an LLM with the label differences between clips drawn from different videos; every pair is thus associated with multiple valid target videos (3.9 on average), reflecting real-world tasks such as sports-highlight generation. To model these temporal dynamics we propose **TF-CoVR-Base**, a concise two-stage training framework: (i) pre-train a video encoder on fine-grained action classification to obtain temporally discriminative embeddings; (ii) align the composed query with candidate videos using contrastive learning. We conduct the first comprehensive study of image, video, and general multimodal embedding (GME) models on temporally fine-grained composed retrieval in both zero-shot and fine-tuning regimes. On TF-CoVR, TF-CoVR-Base improves zero-shot mAP@50 from 5.92 (LanguageBind) to 7.51, and after fine-tuning raises the state-of-the-art from 19.83 to 27.22

## Environment Setup
```
cd TF-CoVR/
conda create -n tfcovr python=3.10 -y
conda activate tfcovr
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

## Pretrained weights
Please download our stage 2 pretrained weights from google drive [here](https://drive.google.com/file/d/1SulhomaUi3VkH9Uo5Ce4L6vzxRoZ_mkP/view?usp=sharing).

## Dataset
Please follow the instructions from [DATASET.md](./data/DATASET.md) to access the dataset.

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
