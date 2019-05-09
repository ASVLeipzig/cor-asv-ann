# cor-asv-ann
    OCR post-correction with encoder-attention-decoder LSTMs

## Introduction

This is a tool for automatic OCR _post-correction_ (reducing optical character recognition errors) with recurrent neural networks. It uses sequence-to-sequence transduction on the _character level_ with a model architecture akin to neural machine translation, i.e. a stacked **encoder-decoder** network with attention mechanism. 

The **attention model** always applies for full lines (in a _global_ configuration), and uses a linear _additive_ alignment model. (This transfers information between the encoder and decoder hidden layer states, and calculates a _soft alignment_ between input and output characters. It is imperative for character-level processing, because with a simple final-initial transfer, models tend to start "forgetting" the input altogether at some point in the line and behave like unconditional LM generators.)

...mention: 
- stacked architecture (with bidirectional bottom and attentional top), configurable depth/width
- weight tying
- underspecification and gap
- confidence input and alternative input
- CPU/GPU option
- A* beam search or soft decoding
- incremental training, LM transfer, shallow transfer
- evaluation (CER, PPL)

When applied on PAGE-XML (as OCR-D workspace processor), this component also allows processing below the `TextLine` hierarchy level, i.e. on `Word` or `Glyph` level. For that it uses the soft alignment scores to calculate an optimal hard alignment path for characters, and thereby distributes the transduction onto the lower level elements (keeping their coordinates and other meta-data), while changing Word segmentation if necessary.

...

### Architecture

...

## Installation

Required Ubuntu packages:

* Python (``python`` or ``python3``)
* pip (``python-pip`` or ``python3-pip``)
* virtualenv (``python-virtualenv`` or ``python3-virtualenv``)

Create and activate a virtualenv as usual.

To install Python dependencies and this module, then do:
```shell
make deps install
```
Which is the equivalent of:
```shell
pip install -r requirements.txt
pip install -e .
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
  "tools": {
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
  }
```

...

## Testing

not yet!
...
