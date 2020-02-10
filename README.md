# cor-asv-ann
    OCR post-correction with encoder-attention-decoder LSTMs

[![CircleCI](https://circleci.com/gh/ASVLeipzig/cor-asv-ann.svg?style=svg)](https://circleci.com/gh/ASVLeipzig/cor-asv-ann)

## Introduction

This is a tool for automatic OCR _post-correction_ (reducing optical character recognition errors) with recurrent neural networks. It uses sequence-to-sequence transduction on the _character level_ with a model architecture akin to neural machine translation, i.e. a stacked **encoder-decoder** network with attention mechanism. 

### Architecture

The **attention model** always applies to full lines (in a _local, monotonic_ configuration), and uses a linear _additive_ alignment model. (This transfers information between the encoder and decoder hidden layer states, and calculates a _soft alignment_ between input and output characters. It is imperative for character-level processing, because with a simple final-initial transfer, models tend to start "forgetting" the input altogether at some point in the line and behave like unconditional LM generators. Local alignment is necessary to prevent snapping back to earlier states during long sequences.)

The **architecture** is as follows: 
0. The input characters are represented as unit vectors (or as a probability distribution in case of uncertainty and ambiguity). These enter a dense projection layer to be picked up by the encoder.
1. The bottom hidden layer of the encoder is a bi-directional LSTM. 
2. The next encoder layers are forward LSTMs stacked on top of each other. 
3. The outputs of the top layer enter the attention model as constants (both in raw form to be weighted with the decoder state recurrently, and in a pre-calculated dense projection).
4. The hidden layers of the decoder are forward LSTMs stacked on top of each other.
5. The top hidden layer of the decoder has double width and contains the attention model:
   - It reads the attention constants from 3. and uses the alignment as attention state (to be input as initial and output as final state). 
   - The attention model masks a window around the center of the previous alignment plus 1 character, calculates a new alignment between encoder outputs and current decoder state, and superimposes this with the encoder outputs to yield a context vector.
   - The context vector is concatenated to the previous layers output and enters the LSTM.
6. The decoder outputs enter a dense projection and get normalized to a probability distribution (softmax) for each character. (The output projection weights are the transpose of the input projection weights in 0. – weight tying.)
7. Depending on the decoder mode, the decoder output is fed back directly (greedy) or indirectly (beamed) into the decoder input. (The first position is fed with a start symbol. Decoding ends on receiving a stop symbol.)
8. The result is the character sequences corresponding to the argmax probabilities of the decoder outputs.

HL depth and width, as well as many other topology and training options can be configured:
- residual connections between layers in encoder and decoder?
- deep bidirectional encoder (with fw/bw cross-summarization)?
- LM loss/prediction as secondary output (multi-task learning, dual scoring)?

### Multi-OCR input

not yet!

### Decoder feedback

One important empirical finding is that the softmax output (full probability distribution) of the decoder can carry important information for the next state when input directly. This greatly improves the accuracy of both alignments and predictions. (This is in part attributable to exposure bias.) Therefore, instead of following the usual convention of feeding back argmax unit vectors, this implementation feeds back the softmax output directly.

This can even be done for beam search (which normally splits up the full distribution into a few select explicit candidates, represented as unit vectors) by simply resetting maximum outputs for lower-scoring candidates successively.

### Decoder modes

While the _encoder_ can always be run in parallel over a batch of lines and by passing the full sequence of characters in one tensor (padded to the longest line in the batch), which is very efficient with Keras backends like Tensorflow, a **beam-search** _decoder_ requires passing initial/final states character-by-character, with parallelism employed to capture multiple history hypotheses of a single line. However, one can also **greedily** use the best output only for each position (without beam search). This latter option also allows to run in parallel over lines, which is much faster – consuming up to ten times less CPU time.

Thererfore, the backend function `lib.Sequence2Sequence.correct_lines` can operate the decoder network in either of the following modes:

#### _fast_

Decode greedily, but feeding back the full softmax distribution in batch mode (lines-parallel).

#### _greedy_

Decode greedily, but feeding back the full softmax distribution for each line separately.

#### _default_

Decode beamed, selecting the best output candidates of the best history hypotheses for each line and feeding back their (successively reset) partial softmax distributions in batch mode (hypotheses-parallel). More specifically:

> Start decoder with start-of-sequence, then keep decoding until
> end-of-sequence is found or output length is way off, repeatedly.
> Decode by using the best predicted output characters and several next-best
> alternatives (up to some degradation threshold) as next input.
> Follow-up on the N best overall candidates (estimated by accumulated
> score, normalized by length and prospective cost), i.e. do A*-like
> breadth-first search, with N equal `batch_size`.
> Pass decoder initial/final states from character to character,
> for each candidate respectively.

### Rejection

During beam search (default decoder mode), whenever the input and output is in good alignment (i.e. the attention model yields an alignment approximately 1 character after their predecessor's alignment on average), it is possible to estimate the current position in the source string. This input character's predicted output score, when smaller than a given (i.e. variable) probability threshold can be clipped to that minimum. This effectively adds a candidate which _rejects_ correction at that position (keeping the input unchanged).

### Underspecification and gap

Input characters that have not been seen during training must be well-behaved at inference time: They must be represented by a reserved index, and should behave like **neutral/unknown** characters instead of spoiling HL states and predictions in a longer follow-up context. This is achieved by dedicated leave-one-out training and regularization to optimize for interpolation of all known characters. At runtime, the encoder merely shows a warning of the previously unseen character.

The same device is useful to fill a known **gap** in the input (the only difference being that no warning is shown).

### Training

Possibilities:
- incremental training and pretraining (on clean-only text)
- scheduled sampling (mixed teacher forcing and decoder feedback)
- LM transfer (initialization of the decoder weights from a language model of the same topology)
- shallow transfer (initialization of encoder/decoder weights from a model of lesser depth)

### Processing PAGE annotations

When applied on PAGE-XML (as OCR-D workspace processor), this component also allows processing below the `TextLine` hierarchy level, i.e. on `Word` or `Glyph` level. For that it uses the soft alignment scores to calculate an optimal hard alignment path for characters, and thereby distributes the transduction onto the lower level elements (keeping their coordinates and other meta-data), while changing Word segmentation if necessary.

...

### Evaluation

Text lines can be compared (by aligning and computing a distance under some metric) across multiple inputs. (This would typically be GT and OCR vs post-correction.) This can be done both on plain text files (`cor-asv-ann-eval`) and PAGE-XML annotations (`ocrd-cor-asv-ann-evaluate`). 

Distances are accumulated (as micro-averages) as character error rate (CER) mean and stddev, but only on the character level.

There are a number of distance metrics available (all operating on grapheme clusters, not mere codepoints):
- `Levenshtein`:  
  simple unweighted edit distance (fastest, standard; GT level 3)
- `NFC`:  
  like `Levenshtein`, but apply Unicode normal form with canonical composition before (i.e. less than GT level 2)
- `NFKC`:  
  like `Levenshtein`, but apply Unicode normal form with compatibility composition before (i.e. less than GT level 2, except for `ſ`, which is already normalized to `s`)
- `historic_latin`:  
  like `Levenshtein`, but decomposing non-vocalic ligatures before and treating as equivalent (i.e. zero distances) confusions of certain semantically close characters often found in historic texts (e.g. umlauts with combining letter `e` as in `Wuͤſte` instead of  to `Wüſte`, `ſ` vs `s`, or quotation/citation marks; GT level 1)

...perplexity measurement...

## Installation

Required Ubuntu packages:

* Python (``python`` or ``python3``)
* pip (``python-pip`` or ``python3-pip``)
* venv (``python-venv`` or ``python3-venv``)

Create and activate a virtual environment as usual.

To install Python dependencies:
```shell
make deps
```
Which is the equivalent of:
```shell
pip install -r requirements.txt
```

To install this module, then do:
```shell
make install
```
Which is the equivalent of:
```shell
pip install .
```

The module can use CUDA-enabled GPUs (when sufficiently installed), but can also run on CPU only. Models are always interchangable.

## Usage

This packages has the following user interfaces:

### command line interface `cor-asv-ann-train`

To be used with string arguments and plain-text files.

...

### command line interface `cor-asv-ann-eval`

To be used with string arguments and plain-text files.

...

### command line interface `cor-asv-ann-repl`

interactive, visualization

...

### [OCR-D processor](https://ocr-d.github.io/cli) interface `ocrd-cor-asv-ann-process`

To be used with [PageXML](https://github.com/PRImA-Research-Lab/PAGE-XML) documents in an [OCR-D](https://ocr-d.github.io/) annotation workflow. 

Input could be anything with a textual annotation (`TextEquiv` on the given `textequiv_level`). 

Pretrained model files are contained in the [models subrepository](models/README.md). At runtime, you can use both absolute and relative paths for model files. The latter are searched for in the installation directory, and under the path in the environment variable `CORASVANN_DATA` (if given).


```json
    "ocrd-cor-asv-ann-process": {
      "executable": "ocrd-cor-asv-ann-process",
      "categories": [
        "Text recognition and optimization"
      ],
      "steps": [
        "recognition/post-correction"
      ],
      "description": "Improve text annotation by character-level encoder-attention-decoder ANN model",
      "input_file_grp": [
        "OCR-D-OCR-TESS",
        "OCR-D-OCR-KRAK",
        "OCR-D-OCR-OCRO",
        "OCR-D-OCR-CALA",
        "OCR-D-OCR-ANY"
      ],
      "output_file_grp": [
        "OCR-D-COR-ASV"
      ],
      "parameters": {
        "model_file": {
          "type": "string",
          "format": "uri",
          "content-type": "application/x-hdf;subtype=bag",
          "description": "path of h5py weight/config file for model trained with cor-asv-ann-train",
          "required": true,
          "cacheable": true
        },
        "textequiv_level": {
          "type": "string",
          "enum": ["line", "word", "glyph"],
          "default": "glyph",
          "description": "PAGE XML hierarchy level to read/write TextEquiv input/output on"
        },
        "rejection_threshold": {
          "type": "number",
          "format": "float",
          "default": 0.5,
          "description": "minimum probability of the candidate corresponding to the input character in each hypothesis during beam search, helps balance precision/recall trade-off; set to 0 to disable rejection (max recall) or 1 to disable correction (max precision)"
        },
        "relative_beam_width": {
          "type": "number",
          "format": "float",
          "default": 0.2,
          "description": "minimum fraction of the best candidate's probability required to enter the beam in each hypothesis; controls the quality/performance trade-off"
        },
        "fixed_beam_width": {
          "type": "number",
          "format": "integer",
          "default": 15,
          "description": "maximum number of candidates allowed to enter the beam in each hypothesis; controls the quality/performance trade-off"
        },
        "fast_mode": {
          "type": "boolean",
          "default": false,
          "description": "decode greedy instead of beamed, with batches of parallel lines instead of parallel alternatives; also disables rejection and beam parameters; enable if performance is far more important than quality"
        }
      }
   }
```

...

### [OCR-D processor](https://ocr-d.github.io/cli) interface `ocrd-cor-asv-ann-evaluate`

To be used with [PageXML](https://github.com/PRImA-Research-Lab/PAGE-XML) documents in an [OCR-D](https://ocr-d.github.io/) annotation workflow.

Inputs could be anything with a textual annotation (`TextEquiv` on the line level), but at least 2. The first in the list of input file groups will be regarded as reference/GT.

There are various evaluation [metrics](#Evaluation) available.

The tool can also aggregate and show the most frequent character confusions.

```json
    "ocrd-cor-asv-ann-evaluate": {
      "executable": "ocrd-cor-asv-ann-evaluate",
      "categories": [
        "Text recognition and optimization"
      ],
      "steps": [
        "recognition/evaluation"
      ],
      "description": "Align different textline annotations and compute distance",
      "input_file_grp": [
        "OCR-D-GT-SEG-LINE",
        "OCR-D-OCR-TESS",
        "OCR-D-OCR-KRAK",
        "OCR-D-OCR-OCRO",
        "OCR-D-OCR-CALA",
        "OCR-D-OCR-ANY",
        "OCR-D-COR-ASV"
      ],
      "parameters": {
        "metric": {
          "type": "string",
          "enum": ["Levenshtein", "NFC", "NFKC", "historic_latin"],
          "default": "Levenshtein",
          "description": "Distance metric to calculate and aggregate: historic_latin for GT level 1, NFKC for GT level 2 (except ſ-s), Levenshtein for GT level 3"
        },
        "confusion": {
          "type": "number",
          "format": "integer",
          "minimum": 0,
          "default": 0,
          "description": "Count edits and show that number of most frequent confusions (non-identity) in the end."
        }
      }
    }
```

...

## Testing

not yet!
...
