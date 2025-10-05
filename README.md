# CRNN-BirdClef2023
## **Automated Bird Species Identification using CRNN — BirdCLEF 2023**

This repository contains the complete implementation of our research titled
**“Automated Bird Species Identification Using Deep Learning: A CRNN Approach to Analysing Avian Vocalizations.”**

The project focuses on **bioacoustic classification** using **Convolutional Recurrent Neural Networks (CRNNs)** trained on the **BirdCLEF 2023** dataset. The model identifies avian species from real-world soundscapes with high accuracy and ecological relevance.

---

### **Repository Structure**

```
.
├── BirdClef2023.ipynb     # Main Jupyter Notebook (complete pipeline)
└── README.md              # Project description and usage instructions
```

---

### **Overview**

* **Goal:** Automate bird species identification using deep learning on soundscape data.
* **Dataset:** BirdCLEF 2023 (Cornell Lab of Ornithology, K.M.H. Centre for Ornithology, Kenya).
* **Model:** CRNN combining CNN feature extraction + BiGRU temporal modeling + attention pooling.
* **Framework:** TensorFlow 2.12 / Keras.
* **Hardware Used:** NVIDIA A100 (40 GB), Intel Xeon 2.20 GHz, 32 GB RAM (Google Colab Pro+).
* **Performance:**

  * **Patch-level:** Accuracy = 90.16 %, Macro-F1 = 0.8723 , pcmAP = 0.9237
  * **Clip-level (Mean):** Accuracy = 91.63 %, Macro-F1 = 0.9084 , pcmAP = 0.9444

---

###  **Features**

* **Log-mel spectrogram** segmentation and preprocessing (10 s segments @ 32 kHz).
* **SpecAugment** and **MixUp** for robust data augmentation.
* **Balanced mini-batch sampling** to mitigate class imbalance.
* **Attention pooling** for interpretable clip-level predictions.
* Evaluation using **pcmAP**, **Macro-F1**, and **Accuracy** (patch & clip levels).

---

### **How to Run**

1. Open the notebook in Google Colab or Jupyter Lab:

   ```
   BirdClef2023.ipynb
   ```
2. Upload your dataset or mount Google Drive as instructed in the notebook.
3. Run each cell sequentially to:

   * Preprocess audio → generate log-mel features.
   * Train the CRNN model.
   * Evaluate using patch-level and clip-level metrics.

> **Tip:** You can modify hyperparameters like `SEG_DUR`, `N_MELS`, and learning rate directly in the notebook.

---

###**Results Summary**

| Level               |   Accuracy  |  Macro-F1  |    pcmAP   |
| :------------------ | :---------: | :--------: | :--------: |
| Patch               |   90.16 %   |   0.8723   |   0.9237   |
| Clip (Mean Pooling) | **91.63 %** | **0.9084** | **0.9444** |
| Clip (Max Pooling)  |   91.15 %   |   0.9029   |   0.8983   |

---

### **Dataset Reference**

> Holger Klinck, Sohier Dane, Stefan Kahl, and Tom Denton.
> **BirdCLEF 2023.** [Kaggle Competition](https://www.kaggle.com/competitions/birdclef-2023).
> Cornell Lab of Ornithology, 2023.

---

### **Citation**

If you use this code or its results, please cite:

> Vijay H. Kalmani, *et al.*
> **“Automated Bird Species Identification Using Deep Learning: A CRNN Approach to Analysing Avian Vocalizations.”**
> *Submitted to Ostrich Journal (2025).*

---

###  **Acknowledgements**

* **Cornell Lab of Ornithology** for providing BirdCLEF datasets.
* **Google Colab Pro+** for computational resources.
* **Reviewers of *Ostrich*** for valuable feedback improving the methodology.

---

###  **Requirements**

To install all dependencies manually:

```bash
pip install tensorflow==2.12.0 numpy pandas librosa matplotlib tqdm
```

---

### 📬 **Contact**

or questions or collaboration:  
**Dr Vijay H. Kalmani**  
*Email:* vijaykalmani@gmail.com 

