# September 21: training with the latest Pip export

Imported `pip-labeled-photos-2026-09-21.json` from Downloads.

## Photo review

- Accepted 9 photos: 7 cups and 2 plates.
- Excluded `Screenshot 2026-09-21 152615.png`, an ambiguous blue high-rim dish, as confirmed by the user. Its original plate label is preserved in `excluded-photos.json`.
- Recomputed SHA-256 hashes and checked against all original train, validation, and test images. No accepted photo was an exact duplicate. Visual review found no obvious copies of held-out photos; this is not an exhaustive near-duplicate guarantee.
- Kept the original duplicate cup image out of training.
- Training now has 89 unique photos: 15 bowls, 48 cups, 26 plates. Validation remains 25 photos; test remains 19.

## Training and comparison

The original 80-image model was reproduced exactly (maximum weight difference: 0). The candidate was trained from scratch with the same 48-pixel color features, 600 passes, learning rate 0.5, and L2 0.01. Grayscale was not introduced. Probability temperature was selected on validation data and remained 0.75.

Before examining test results, the candidate was rejected for promotion using validation macro recall, with calibrated validation loss as the tie-breaker. This is a fixed-settings experiment, not a new hyperparameter search.

| Check | Current model | Retrained candidate |
| --- | --- | --- |
| Validation correct | 22/25 | 22/25 |
| Validation macro recall | 86.1% | 86.1% |
| Validation loss (lower is better) | 0.6220 | 0.6474 |
| Test correct | 13/19 (68%) | 11/19 (58%) |
| Added photos correct | 2/9 | 8/9 |

The added-photo result measures learning on training examples, not performance on unseen photos. The existing test set is small and has been checked before; a fresh independent test set would provide stronger evidence.

## Outcome and files

The main `dishware-classifier.html` and its model are unchanged. Open `../../dishware-classifier-retrained.html` to try the experimental model. Its recorded training display uses the new weights and the correct 89-image count.

- `training-data.json`: accepted dataset records with original split, label, source, and SHA-256; image paths are relative to this folder.
- `images/`: original image bytes, including the separately excluded photo for traceability.
- `import-review.json` and `excluded-photos.json`: review decisions.
- `candidate-model.json`: retrained weights and settings.
- `candidate-training-history.json`: actual recorded checkpoints.
- `results.json`: full comparison and per-added-photo predictions.
- `previous-*.json`: previous model, history, and experiment results.
- `retrain.js`, `run-training.html`, `review-config.json`: reproducible experiment. Serve this folder locally and open `run-training.html`; the page fetches local JSON and images and displays its result as JSON. Local file execution requires a browser configured to allow local-file access.

The original export in Downloads was not changed. No photos were sent to a server.