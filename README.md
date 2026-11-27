# Handwritten notes OCR

An archived prototype that explores converting handwritten word images to text. The TensorFlow experiment uses convolutional features, bidirectional LSTMs, and CTC decoding. The repository is a research snapshot, not a packaged OCR app or a reproducible benchmark.

## Where to start

- [HandwrittenRecog.py](HandwrittenRecog.py) covers the data pipeline, model, checkpoint restoration, and single-image decoding.
- [HandwrittenRecogNB.ipynb](HandwrittenRecogNB.ipynb) preserves the notebook experiment. [KerasOCR.ipynb](KerasOCR.ipynb) is a separate OCR experiment.
- [Checkpoints/](Checkpoints/) contains historical model artifacts. A checked-in artifact is not evidence of a clean end-to-end run.

## Reproducibility limits

The script expects a separate `data/words.txt` index and matching `data/words/` images. Its final example still points to an absolute Windows path from the original checkout. Neither those inputs nor an installation with pinned compatible dependencies is provided as a self-contained demo. A syntax-only inspection that does not load model code is:

```sh
python -m py_compile HandwrittenRecog.py
```

Keep this repo private until rights to the handwriting samples, training artifacts, and notebook outputs have been checked. A public demo should use a small, explicitly shareable image fixture and a separate, documented inference command. Do not unpickle bundled or downloaded artifacts unless you trust their source.
