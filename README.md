# cor-asv-ann
    OCR post-correction with encoder-attention-decoder LSTMs

[![CircleCI](https://circleci.com/gh/ASVLeipzig/cor-asv-ann.svg?style=svg)](https://circleci.com/gh/ASVLeipzig/cor-asv-ann)

## Introduction

This is a tool for automatic OCR _post-correction_ (reducing optical character recognition errors) with recurrent neural networks. It uses sequence-to-sequence transduction on the _character level_ with a model architecture akin to neural machine translation, i.e. a stacked **encoder-decoder** network with attention mechanism. 

The **attention model** always applies to full lines (in a _global_ configuration), and uses a linear _additive_ alignment model. (This transfers information between the encoder and decoder hidden layer states, and calculates a _soft alignment_ between input and output characters. It is imperative for character-level processing, because with a simple final-initial transfer, models tend to start "forgetting" the input altogether at some point in the line and behave like unconditional LM generators.)

...FIXME: mention: 
- stacked architecture (with bidirectional bottom and attentional top), configurable depth/width
- weight tying
- underspecification and gap
- confidence input and alternative input
- CPU/GPU option
- incremental training, LM transfer, shallow transfer
- evaluation (CER, PPL)

### Processing PAGE annotations

When applied on PAGE-XML (as OCR-D workspace processor), this component also allows processing below the `TextLine` hierarchy level, i.e. on `Word` or `Glyph` level. For that it uses the soft alignment scores to calculate an optimal hard alignment path for characters, and thereby distributes the transduction onto the lower level elements (keeping their coordinates and other meta-data), while changing Word segmentation if necessary.

...

### Architecture

...FIXME: show!

### Input with confidence and/or alternatives

...FIXME: explain!

### Multi-OCR input

not yet!

### Modes

While the _encoder_ can always be run in parallel over a batch of lines and by passing the full sequence of characters in one tensor (padded to the longest line in the batch), which is very efficient with Keras backends like Tensorflow, a **beam-search** _decoder_ requires passing initial/final states character-by-character, with parallelism employed to capture multiple history hypotheses of a single line. However, one can also **greedily** use the best output only for each position (without beam search). And in doing so, another option is to feed back the softmax output directly into the decoder input instead of its argmax unit vector. This effectively passes the full probability distribution from state to state, which (not very surprisingly) can increase correction accuracy quite a lot – it can get as good as a medium-sized beam search results. This latter option also allows to run in parallel again, which is also much faster – consuming up to ten times less CPU time.

Thererfore, the backend function `lib.Sequence2Sequence.correct_lines` can operate the encoder-decoder network in either of the following modes:

#### _fast_

Decode greedily, but feeding back the full softmax distribution in batch mode.

#### _greedy_

Decode greedily, but feeding back the argmax unit vectors for each line separately.

#### _default_

Decode beamed, feeding back the argmax unit vectors for the best history/output hypotheses of each line. More specifically:

> Start decoder with start-of-sequence, then keep decoding until
> end-of-sequence is found or output length is way off, repeatedly.
> Decode by using the best predicted output characters and several next-best
> alternatives (up to some degradation threshold) as next input.
> Follow-up on the N best overall candidates (estimated by accumulated
> score, normalized by length and prospective cost), i.e. do A*-like
> breadth-first search, with N equal `batch_size`.
> Pass decoder initial/final states from character to character,
> for each candidate respectively.
> Reserve 1 candidate per iteration for running through `source_seq`
> (as a rejection fallback) to ensure that path does not fall off the
> beam and at least one solution can be found within the search limits.

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


## Installation

Required Ubuntu packages:

* Python (``python`` or ``python3``)
* pip (``python-pip`` or ``python3-pip``)
* virtualenv (``python-venv`` or ``python3-venv``)

Create and activate a virtualenv as usual.

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

## Usage

This packages has the following user interfaces:

### command line interface `cor-asv-ann-train`

To be used with string arguments and plain-text files.

...

### command line interface `cor-asv-ann-eval`

To be used with string arguments and plain-text files.

...

### command line interface `cor-asv-ann-repl`

interactive

...

### [OCR-D processor](https://github.com/OCR-D/core) interface `ocrd-cor-asv-ann-process`

To be used with [PageXML](https://www.primaresearch.org/tools/PAGELibraries) documents in an [OCR-D](https://github.com/OCR-D/spec/) annotation workflow. Input could be anything with a textual annotation (`TextEquiv` on the given `textequiv_level`). 

...

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
        }
      }
    }
```

...

### [OCR-D processor](https://github.com/OCR-D/core) interface `ocrd-cor-asv-ann-evaluate`

To be used with [PageXML](https://www.primaresearch.org/tools/PAGELibraries) documents in an [OCR-D](https://github.com/OCR-D/spec/) annotation workflow. Inputs could be anything with a textual annotation (`TextEquiv` on the line level), but at least 2. The first in the list of input file groups will be regarded as reference/GT.

...

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
      "parameters": {
        "metric": {
          "type": "string",
          "enum": ["Levenshtein", "NFC", "NFKC", "historic_latin"],
          "default": "Levenshtein",
          "description": "Distance metric to calculate and aggregate: historic_latin for GT level 1, NFKC for GT level 2 (except ſ-s), Levenshtein for GT level 3"
        }
      }
    }
```

...

## Testing

not yet!
...
