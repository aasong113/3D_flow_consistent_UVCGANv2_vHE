# 3D UVC FlowGAN: Virtual H&E Staining with Spatially Consistent Flow Loss

**3D Virtual H&E staining using z-axis flow and gradient loss for spatial consistency.**  
This method translates 3D Back-illumination Interference Tomography (BIT) image volumes to virtually stained H&E without the need for 3D histology—only 2D target FFPE H&E slices are required.

---

### 🔬 Original BIT (Left) → Virtual H&E (Right)

**Fresh human duodenum from Whipple surgery**, imaged with our multimodal BIT microscope.

<p float="left">
  <img src="https://github.com/user-attachments/assets/4ce83eaf-b8b2-43b0-8dec-cf12252ea2c8" width="45%"/>
  <img src="https://github.com/user-attachments/assets/95739260-1845-47b8-ac83-529f14f7abb5" width="45%"/>
</p>

**BIT Input Volume**: ~302×211×30 µm³  
**Virtual H&E Output**: ~302×211×30 µm³

---

### 🔄 3D Virtual Volume Rotation
![Movie_4x_downsampled-ezgif com-optimize](https://github.com/user-attachments/assets/c0ae4fca-9928-4ec3-973a-e079033d1b65)

---

### 🧠 3D Volume Renderings

<p float="left">
  <img src="https://github.com/user-attachments/assets/7cc56886-e8bc-4561-b022-08afe8cfc32c" width="45%" />
  <img src="https://github.com/user-attachments/assets/1b78d323-4936-4aec-93d1-e8eb557f6fe1" width="45%" />
</p>

---

### 🧾 Ground Truth FFPE H&E Patches

[duodenum_crypts_roi_18](https://github.com/user-attachments/files/24579530/duodenum_crypts_OTS-25-25256_2025-08-01_13-43-05_roi_18_00018.tif)  
[duodenum_crypts_roi_11](https://github.com/user-attachments/files/24579535/duodenum_crypts_OTS-25-25256_2025-08-01_13-43-05_roi_11_00024.tif)  
[duodenum_crypts_roi_5](https://github.com/user-attachments/files/24579537/duodenum_crypts_OTS-25-25256_2025-08-01_13-43-05_roi_5_00027.tif)  
[duodenum_crypts_roi_13](https://github.com/user-attachments/files/24579540/duodenum_crypts_OTS-25-25256_2025-08-01_13-43-05_roi_13_00067.tif)

---

### 📜 References

- **Proceedings**: [NTM 2025 - NTh1C.3](https://opg.optica.org/abstract.cfm?URI=NTM-2025-NTh1C.3)  
- PDF: `ntm-2025-nth1c.3.pdf` (see repository)
- **Original architecture inspiration**: [UVCGANv2: Rethinking CycleGAN for Scientific Image Translation][uvcgan2_paper]

---

## 🧠 Overview

`3D UVC FlowGAN` builds on [UVCGANv2](https://github.com/uvcgan/uvcgan2) to enable **unpaired 3D→2D virtual staining** using:

- Enhanced **3D generators** with volumetric convolution
- **Flow and gradient consistency loss** across z-slices
- Only 3D *source domain* (e.g., BIT) required
- Only 2D *target domain* (e.g., FFPE H&E) required

---

## 📁 Dataset Format

Your dataset should follow the format:

```bash
BIT/              # Input 3D volumes
  trainA/
  testA/

FFPE_HE/          # Target 2D H&E images
  trainB/
  testB/
