Used materials:
  -  Selfie Dataset http://crcv.ucf.edu/data/Selfie/
  - Flicker8k_Dataset https://forms.illinois.edu/sec/1713398

Total 15000 training photos, equal class distribution
---
File descriptions:
main_selfie.py - transfer learning models
test_simple_models.py - own simple models
labeller.py - creates csv file with object path and ground-true classes
selfiePredictor.py - evaluates and estimates model on a test set from csv-file