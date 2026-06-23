"""roi_classifier — ROI/Half-Projection Presence Classifier tools (article split).

Port of the head/testis presence classifier onto the article repo's own
120/38/39 split (seed 42).  Provides:

- make_classifier_manifest:  derive per-case presence labels from GT label masks
- audit_roi_orientation:      auto PASS/FAIL orientation check for all 197 cases
- generate_roi_projections:   2D lateral projections + anatomical ROI crops
- train_roi_classifier:       30-model ResNet-18 grid training
- summarize_roi_results:      cross-grid best selection + result card
- eval_test:                  one-shot test evaluation on frozen best models
"""

__version__ = "1.0.0"
