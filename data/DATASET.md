# 📂 TF-CoVR Dataset

As we are **not authorized to share the original videos** from [*FineGym*](https://github.com/SDOlivia/FineGym/) and [*FineDiving*](https://github.com/xujinglin/FineDiving), we instead release the **embeddings** generated during our experiments.  

---

## 📑 Annotations
- Train / Validation / Test CSV files are available on [HuggingFace](https://huggingface.co/datasets/ucf-crcv/TF-CoVR).

---

## 🔗 Embeddings
We provide two sets of embeddings:
1. **Finetuned AIM (12-frame inputs)** – embeddings extracted from our fine-tuned model.  
2. **BLIP-2 embeddings** – for comparison with a strong baseline.  

➡️ Both embedding sets can be downloaded from our [HugginFace repo](https://huggingface.co/datasets/ucf-crcv/TF-CoVR).

---

## 🎥 Working with Original Videos
If you would like to use the **raw videos**, please follow the official dataset repositories and then apply our preprocessing code.  

We provide scripts to generate **sub-action videos** (FineGym) and **video clips from frames** (FineDiving).  

---

### 🏋️ FineGym
1. Download the dataset and annotations from the official [FineGym repo]().  
   - Required file: `finegym_annotation_info_v1.1.json`  
   - This file is necessary to trim long videos into sub-action clips.  

2. **Trim long videos into event-level clips**:
   ```bash
   python trim_events.py
    ```
3. **Trim event clips into sub-actions:**
   ```bash
   python trim_subactions.py
   ```

### 🏊 FineDiving
1. Download the dataset from the official FineDiving repo.

2. Once frames are available, convert them into videos using:
    ```bash
    python frames_videos.py
    ```