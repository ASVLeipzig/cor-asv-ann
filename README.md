# cor-asv-fst
    OCR post-correction with attention encoder-decoder LSTMs

## Introduction


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

This packages has two user interfaces:

### command line interface `cor-asv-ann-train`

To be used with string arguments and plain-text files.

...

### command line interface `cor-asv-ann-eval`

To be used with string arguments and plain-text files.

...

### [OCR-D processor](https://github.com/OCR-D/core) interface `ocrd-cor-asv-ann-process`

To be used with [PageXML](https://www.primaresearch.org/tools/PAGELibraries) documents in an [OCR-D](https://github.com/OCR-D/spec/) annotation workflow. Input could be anything with a textual annotation (`TextEquiv` on the given `textequiv_level`). 

...

```json
  "tools": {
    "cor-asv-ann-process": {
      "executable": "cor-asv-ann-process",
      "categories": [
        "Text recognition and optimization"
      ],
      "steps": [
        "recognition/post-correction"
      ],
      "description": "Improve text annotation by character-level attention-based encoder-decoder LSTM neural model",
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
          "description": "path of h5py weight/config file for model from cor-asv-ann-train",
          "required": true,
          "cacheable": true
        },
        "textequiv_level": {
          "type": "string",
          "enum": ["word", "glyph"],
          "default": "glyph",
          "description": "PAGE XML hierarchy level to read TextEquiv input on (output will always be word level)"
        },
        "beam_width": {
          "type": "number",
          "format": "integer",
          "description": "maximum number of best partial paths to consider during beam search in language modelling",
          "default": 100
        }
      }
    }
  }
```

...

## Testing

...
