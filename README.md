### **1. `Experimentation.ipynb`**
**Reproduction of the original Attention-based U-Net paper**

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
