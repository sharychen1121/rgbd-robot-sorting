# Learning pipeline integration check

This is a small functional check, not a finished accuracy study or proof that mask-based sorting is better. It verifies data generation, five CPU fine-tuning epochs of YOLO26n-seg, explicit held-out test validation, checkpoint reload, predicted masks, and mask/box RGB-D localization.

Dataset: 90 independently seeded 320 x 240 scenes, randomized colors and lighting; 62 train / 13 validation / 15 test images, including 14 explicitly empty scenes. Training: 5 epochs, image size 320, batch 2, CPU, seed 241114.

Held-out test box mAP50: 0.2647; mask mAP50: 0.1931; mask mAP50-95: 0.0614. These low pilot metrics indicate that the model needs substantially more training and evaluation.

The unseen image uses scene seed 241115, with 6 candidates at diagnostic confidence 0.10 and 0 at the default confidence 0.25. The diagnostic candidates include duplicate detections and incorrect class predictions; a valid 3D point only means that a predicted region can be localized, not that its category or grasp is correct. The runtime camera frame contains no simulator segmentation buffer.

Full metrics, checkpoint SHA256, model versions, configuration, and candidate details are in learning-smoke.json. The checkpoint is a pilot artifact and should not be presented as the final project model.
