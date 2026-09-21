# Pip: model, learning evidence, and visitor labels

Open `dishware-classifier.html` in Chrome or Edge. The HTML contains everything it needs: the model, original test images, illustrations, and recorded training history. No API key, paid service, server, or build step is required.

## What the classifier learned

Pip is a small linear softmax classifier. It uses 384 hand-designed measurements of shading, colors, and edges, and learns 1,155 weights and biases. It is not a deep image-recognition network.

The selected model trained for 600 passes on 80 unique photos: 15 bowls, 41 cups, and 24 plates. One duplicate cup file was excluded. Each pass compares predicted probabilities with the correct labels and updates the weights through gradient descent. Class-balanced cross-entropy compensates for the unequal class sizes; L2 regularization discourages excessively large weights.

Settings: learning rate 0.5, L2 0.01, no augmentation, input size 48 x 48. Probability temperature is 0.75, selected on the validation set. Twenty-five validation photos are too few to establish trustworthy confidence calibration.

## Real training history

The bottom toggle, "Pip's practice notebook", contains a recorded run, not a simulated learning animation. It includes a replay slider, loss chart, checkpoint table, three actual weights, and a downloadable JSON log. Replaying the log does not train a new model or use visitor contributions.

The old 24-pixel model was reproduced exactly before the resolution comparison (maximum weight difference: zero). The sample weights at the end of the new log are checked against the weights embedded in the page.

- Class-balanced training loss: 1.098612 at pass 0 -> 0.442892 at pass 600.
- Final training result: 70 / 80.
- Final validation result: 22 / 25.
- Three example final weights: -0.031324 (Bowl shading), -0.050215 (Cup edge), +0.421854 (Plate shading).
- All 1,155 weights and biases start at zero. Initial tied predictions default to Bowl; that tie rule explains the initial accuracy.

The log records checkpoints at pass 0, pass 1, and every 25 passes. "Update size" is the Euclidean norm of the weight update in that one pass, not the change since the preceding displayed checkpoint. Training loss is class-balanced; validation loss is unweighted. Neither is the post-temperature-calibration loss.

Supporting files are in `pip-art`: `recorded-training-log.json`, `training-results.json`, and `training-experiment.js`. The experiment script expects image records in a `data` JSON script element and a `report` element, as documented at the top of the script.

## Whole-image processing: no cropping

The old and new versions both preserve the entire image. They fit the photo into a square, retain its aspect ratio, and add white padding. Small handles and edges can be lost through downsampling, not because the page crops them out.

We compared 24, 48, and 72 pixels using original training and validation images. For each size, candidate L2 settings were 0.001/0.01/0.1 and candidate checkpoints were 100/300/600/1000 passes. Selection used average per-class validation accuracy, breaking ties by validation loss. The test set was evaluated after selection; it did not choose the model. The selected model uses 48 pixels. A larger input alone is not an accuracy guarantee.

| Test class | Earlier 24-pixel learned model | Selected 48-pixel model |
| --- | --- | --- |
| Bowl | 4 / 6 | 4 / 6 |
| Cup | 4 / 8 | 6 / 8 |
| Plate | 3 / 5 | 3 / 5 |
| Total | 11 / 19 (58%) | 13 / 19 (68%) |

Each test photo changes the percentage by about 5.3 points. This small improvement is not proof of reliable accuracy on new dishes. Small datasets, background differences, and simple hand-designed features remain important limitations. Training/validation settings have been explored using the same small dataset; collecting a fresh independent test set would give stronger evidence.

Built-in test images are embedded byte-for-byte from the originals. Their thumbnail display size does not affect model input. The previous 63% figure came from inconsistent preprocessing and is obsolete.

## Visitor labels and future retraining

1. Upload a photo.
2. Choose its real class in "What is your dish, really?".
3. Press "Save photo + label on this device".
4. Use "Export labeled photos" to download the collection for later review and retraining.

The page uses IndexedDB to store original image bytes, the chosen label, original filename/type/dimensions, timestamps, a random local contributor-profile identifier, model version, original predicted probabilities, and review flags. It does not send photos to a server. Only pressing Save stores an uploaded photo; simply making a prediction does not.

The visible counters report contributor profiles and unique labeled images stored in this browser. Profiles are not verified humans, and the count is not a site-wide visitor count. One browser typically represents one profile. There is no shared database across devices. Exporting lets you collect files from different devices for future aggregation. Renaming or moving a local HTML file, changing browsers, clearing storage, or using private browsing can affect access to locally saved data; keep exports as backups.

Duplicate photos are detected by SHA-256 of their original bytes. Relabeling the same photo updates its record rather than increasing the image count. Exact matches to the 19 reserved test photos are flagged `reservedTest: true` and `excludeFromTraining: true`. This does not detect edited or recompressed versions, so review the exported collection for near-duplicates too.

Exports include original images as data URLs plus labels and metadata. All labels have `reviewStatus: unreviewed`. Review labels before future retraining; keep reserved test examples out of training. Unrelated or ambiguous images can receive a dishware prediction because Pip only has three output classes.

Saving labels does not update Pip's weights or claim an accuracy improvement. The contributions are candidates for a future training run. Shared site-wide counts and automatic collection would require adding a backend, which this standalone page does not have.

## Pip's expressions

- Celebrating: top probability >= 80%, leading the runner-up by >= 30 percentage points.
- Very confused: top probability < 50%, or the lead is < 12 percentage points.
- Wondering: all other predictions.

These indicate model confidence, not correctness. The updated hero illustration shows three empty shelves and a cardboard box of dishes in front of stairs. All illustrations are embedded in the HTML; source PNGs and prompts are in `pip-art`.

Dataset: https://www.kaggle.com/datasets/adhamabdelazeem/dishware-classification-dataset