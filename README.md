## Additional Dataset: landcover.ai

For the extended experiments in **my context**, I incorporate a new publicly available land-cover dataset:

**landcover.ai** – https://landcover.ai.linuxpolska.com/#fnref:1

For the extended experiments on the **landcover.ai** dataset,  
all data preprocessing is implemented in the script:

`preprocess-extra-rgb-data.py`

This script prepares the landcover.ai images and masks (normalization, resizing, label encoding, and train/val split) so that they can be used directly in the notebooks `Experimentation_extra_datasets-5classes.ipynb` and `Experimentation_extra_datasets-5classes-plus2.ipynb`.
### **1. `Experimentation.ipynb`**
**Reproduction of the original Attention-based U-Net paper**
Paper:"An attention-based U-Net for detecting deforestation within satellite sensor imagery"
Code Link: https://github.com/davej23/attention-mechanism-unet
This notebook contains:

- The full implementation of the Attention U-Net architecture.
- Fixes to bugs in the original authors’ public code.
- Adjusted training loops and correct handling of data dimensions.
- Appropriate epoch settings for each model to ensure convergence.
- Training and validation on the Amazon and Atlantic Forest datasets.
- Evaluation metrics and visual results.

> **Note:** All model weight files (e.g., `.hdf5`, `.npy`) are omitted because they exceed GitHub’s limit.

---

### **2. `Experimentation_extra_datasets-5classes-plus2.ipynb`**
**My improved Attention U-Net on a new dataset (`landcover.ai`)**

This notebook includes:

- The **“my context” optimized version of Attention U-Net**, where several architectural and training refinements were added.
- Experiments performed on a **new 5-class landcover dataset**.
- Extended evaluation metrics and comparative results.
- Demonstration of improved generalization compared to the original model.

---

### **3. `Experimentation_extra_datasets-5classes.ipynb`**
**Original Attention U-Net applied to `landcover.ai` (baseline comparison)**

This notebook contains:

- A baseline experiment applying the *original* Attention U-Net architecture to the same `landcover.ai` dataset.
- Provides direct comparison with the optimized version from the “plus2” notebook.
- Includes performance metrics, qualitative predictions.

---
