# Disclaimer

## Research-Only Warning

This project is a research prototype and not a clinical diagnostic tool.

It is intended for education, experimentation, portfolio demonstration, and medical AI engineering exploration.

## Not a Clinical Diagnosis

The system does not diagnose cancer, rule out disease, recommend treatment, or replace clinical judgment.

Outputs such as:

- Predicted class
- Confidence score
- Grad-CAM heatmap
- Pseudo-3D attention volume
- AI clinical support note
- Chatbot responses

must not be interpreted as medical diagnosis.

## Requires Radiologist or Physician Review

All results require review by a qualified radiologist, physician, or appropriate healthcare professional.

Clinical interpretation requires the full imaging study, patient history, symptoms, prior scans, and appropriate diagnostic workup.

## Dataset Limitations

The dataset used by this project may not represent all scanners, hospitals, demographics, disease stages, acquisition protocols, or real-world clinical variation.

Potential limitations include:

- Class imbalance
- Limited sample diversity
- 2D image slices rather than full volumetric CT studies
- Dataset-specific artifacts
- Unknown external generalization

## Explainability Limitations

Grad-CAM and pseudo-3D attention visualizations show where the model focused. They do not prove tumor location, boundaries, severity, or diagnosis.

Attention maps can be influenced by artifacts, preprocessing, dataset bias, and model uncertainty.

## Clinical Use Boundary

Do not use this system for patient care unless it has undergone proper validation, regulatory review, clinical governance, and professional oversight.
