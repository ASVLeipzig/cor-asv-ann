# pylint: disable=import-error

import os
import json
import numpy as np
import pytest

from ocrd import run_processor
from ocrd_utils import MIMETYPE_PAGE
from ocrd_models.constants import NAMESPACES as NS
from ocrd_modelfactory import page_from_file

from ocrd_cor_asv_ann.wrapper.transcode import ANNCorrection
from ocrd_cor_asv_ann.wrapper.mark import MarkWords
from ocrd_cor_asv_ann.wrapper.join import JoinLines
from ocrd_cor_asv_ann.wrapper.align import AlignLines
from ocrd_cor_asv_ann.wrapper.evaluate import EvaluateLines

MODEL = 'models/s2s.dta19.Fraktur4.d2.w0512.adam.attention.stateless.variational-dropout.char.pretrained+retrained-conf.h5'

def test_transcode(workspace, subtests, record_property):
    ws, page_id = workspace
    pages = page_id.split(',')
    page1 = pages[0]
    grps = [grp for grp in ws.mets.file_groups
            if 'OCR-D-OCR-' in grp and (
                    # above PC model does not work well with models based on GT4HistOCR transcriptions
                    # (need to re-train)
                    not 'gt4histocr' in grp and
                    # also, without GPU these combinations take too long, so skip more variants
                    not 'OCRO' in grp and
                    not 'Latin' in grp and
                    not 'deu' in grp)]
    for input_file_grp in grps:
        inputs = list(ws.find_files(file_grp=input_file_grp, mimetype=MIMETYPE_PAGE))
        iconfs = [float(conf) for input in inputs
                  for conf in page_from_file(input).etree.xpath(
                          "//page:TextLine/page:TextEquiv/@conf", namespaces=NS)]
        iconf0 = np.average(iconfs)
        assert 0 < iconf0 < 1.0
        record_property('avg-confidence-input', iconf0)
        for textequiv_level in ['glyph', 'word', 'line']:
            for fast_mode in [True, False]:
                output_file_grp = input_file_grp + '-PC-ANN-' + textequiv_level + ('-fast' if fast_mode else '')
                with subtests.test(
                        msg="test post-correction",
                        input_file_grp=input_file_grp,
                        textequiv_level=textequiv_level,
                        fast_mode=fast_mode):
                    run_processor(
                        ANNCorrection,
                        input_file_grp=input_file_grp,
                        output_file_grp=output_file_grp,
                        parameter=dict(textequiv_level=textequiv_level,
                                       model_file=MODEL,
                                       rejection_threshold=0.1,
                                       fixed_beam_width=50,
                                       fast_mode=fast_mode),
                        workspace=ws,
                        page_id=page_id,
                    )
                    ws.save_mets()
                    assert os.path.isdir(os.path.join(ws.directory, output_file_grp))
                    results = list(ws.find_files(file_grp=output_file_grp, mimetype=MIMETYPE_PAGE))
                    assert len(results), "found no output PAGE files"
                    assert len(results) == len(pages)
                    result1 = results[0]
                    assert os.path.exists(result1.local_filename), "result for first page not found in filesystem"
                    oconfs = [float(conf) for result in results
                              for conf in page_from_file(result).etree.xpath(
                                      "//page:TextLine/page:TextEquiv/@conf", namespaces=NS)]
                    oconf0 = np.average(oconfs)
                    assert 0 < oconf0 < 1.0
                    assert iconf0 != oconf0
                    record_property('avg-confidence-output', oconf0)
                    # if fast_mode:
                    #     # bad quality
                    #     continue
                    tree1 = page_from_file(result1).etree
                    line1 = tree1.xpath(
                        "//page:TextLine[page:TextEquiv/page:Unicode[contains(text(),'Aufklärung') or contains(text(),'Aufklaͤrung')]]",
                        namespaces=NS,
                    )
                    assert len(line1) >= 1, "result is inaccurate"
                    line1 = line1[0]
                    line1_text = line1.xpath("page:TextEquiv[1]/page:Unicode/text()", namespaces=NS)[0]
                    words = line1.xpath(".//page:Word", namespaces=NS)
                    if textequiv_level == 'line':
                        assert len(words) == 0
                        continue
                    # The textline should
                    # a. contain multiple words and
                    # b. these should concatenate fine to produce the same line text
                    assert len(words) > 2, "result does not contain words"
                    words_text = " ".join(
                        word.xpath("page:TextEquiv[1]/page:Unicode/text()", namespaces=NS)[0]
                        for word in words
                    )
                    assert words_text == line1_text, "word-level text result does not concatenate to line-level text result"
                    glyphs = line1.xpath(".//page:Glyph", namespaces=NS)
                    if textequiv_level == 'word':
                        assert len(glyphs) == 0
                        continue
                    assert len(glyphs) > 15, "result must contain glyphs"
    
def test_all(processor_kwargs, subtests, record_property):
    ws = processor_kwargs['workspace']
    pages = processor_kwargs['page_id'].split(',')
    page1 = pages[0]
    input_file_grp = 'OCR-D-OCR-TESS-Fraktur-SEG-LINE-tesseract-ocropy-DEWARP'
    output_file_grp = 'OCR-D-OCR-PC-ANN'
    with subtests.test(msg="test post-correction"):
        run_processor(ANNCorrection,
                      input_file_grp=input_file_grp,
                      output_file_grp=output_file_grp,
                      parameter={
                          'textequiv_level': 'glyph',
                          'model_file': MODEL,
                          'rejection_threshold': 0.1,
                          'fixed_beam_width': 50,
                      },
                      **processor_kwargs,
        )
        ws.save_mets()
        assert os.path.isdir(os.path.join(ws.directory, output_file_grp))
        results = list(ws.find_files(file_grp=output_file_grp, mimetype=MIMETYPE_PAGE))
        assert len(results), "found no output PAGE files"
        assert len(results) == len(pages)
        result1 = results[0]
        assert os.path.exists(result1.local_filename), "result for first page not found in filesystem"
        tree1 = page_from_file(result1).etree
        line1 = tree1.xpath(
            "//page:TextLine[page:TextEquiv/page:Unicode[contains(text(),'Aufklärung')]]",
            namespaces=NS,
        )
        assert len(line1) >= 1, "result is inaccurate"
        line1 = line1[0]
        line1_text = line1.xpath("page:TextEquiv[1]/page:Unicode/text()", namespaces=NS)[0]
        # The textline should
        # a. contain multiple words and
        # b. these should concatenate fine to produce the same line text
        words = line1.xpath(".//page:Word", namespaces=NS)
        assert len(words) > 2, "result does not contain words"
        words_text = " ".join(
            word.xpath("page:TextEquiv[1]/page:Unicode/text()", namespaces=NS)[0]
            for word in words
        )
        assert words_text == line1_text, "word-level text result does not concatenate to line-level text result"
        glyphs = line1.xpath(".//page:Glyph", namespaces=NS)
        assert len(glyphs) > 15, "result must contain glyphs"

    grps = [grp for grp in ws.mets.file_groups
            if 'OCR-D-OCR-' in grp]
    input_file_grp = ','.join(grps)
    output_file_grp = 'OCR-D-OCR-MULTI'
    with subtests.test(msg="test align"):
        run_processor(AlignLines,
                      input_file_grp=input_file_grp,
                      output_file_grp=output_file_grp,
                      parameter={
                          'method': 'combined',
                      },
                      **processor_kwargs,
        )
        ws.save_mets()
        assert os.path.isdir(os.path.join(ws.directory, output_file_grp))
        results = list(ws.find_files(file_grp=output_file_grp, mimetype=MIMETYPE_PAGE))
        assert len(results), "found no output PAGE files"
        assert len(results) == len(pages)
        result1 = results[0]
        assert os.path.exists(result1.local_filename), "result for first page not found in filesystem"
        tree1 = page_from_file(result1).etree
        line1 = tree1.xpath(
            "//page:TextLine[page:TextEquiv/page:Unicode[contains(text(),'Aufklärung')]]",
            namespaces=NS,
        )
        assert len(line1) >= 1, "result is inaccurate"
        line1 = line1[0]

    # use OCR-D-OCR-MULTI as reference (because our GT has different segmentation)
    input_file_grp = ','.join([output_file_grp] + grps)
    output_file_grp = 'OCR-D-EVAL'
    with subtests.test(msg="test evaluate (multiple fileGrps)"):
        run_processor(EvaluateLines,
                      input_file_grp=input_file_grp,
                      output_file_grp=output_file_grp,
                      parameter={
                          'match_on': 'id',
                          'confusion': 10,
                          'histogram': True,
                          'metric': 'historic_latin',
                          'gt_level': 2,
                      },
                      **processor_kwargs,
        )
        ws.save_mets()
        assert os.path.isdir(os.path.join(ws.directory, output_file_grp))
        results = list(ws.find_files(file_grp=output_file_grp, mimetype='application/json'))
        assert len(results), "found no output JSON report files"
        assert len(results) == len(pages) + 1
        result0 = next((result for result in results if result.pageId is None), None)
        assert result0 is not None, "found no document-wide JSON report file"
        with open(result0.local_filename) as file:
            result0 = json.load(file)
        for ocr_file_grp in grps:
            assert "%s,OCR-D-OCR-MULTI" % ocr_file_grp in result0
        eval_result = result0

    output_file_grp = 'OCR-D-OCR-ALL'
    with subtests.test(msg="test join"):
        run_processor(JoinLines,
                      input_file_grp=input_file_grp,
                      output_file_grp=output_file_grp,
                      parameter={
                          'match-on': 'id',
                          'add-filegrp-index': True,
                      },
                      **processor_kwargs,
        )
        ws.save_mets()
        assert os.path.isdir(os.path.join(ws.directory, output_file_grp))
        results = list(ws.find_files(file_grp=output_file_grp, mimetype=MIMETYPE_PAGE))
        assert len(results), "found no output PAGE files"
        assert len(results) == len(pages)
        result1 = results[0]
        assert os.path.exists(result1.local_filename), "result for first page not found in filesystem"
        tree1 = page_from_file(result1).etree
        line1 = tree1.xpath(
            "//page:TextLine[page:TextEquiv/page:Unicode[contains(text(),'Aufklaͤrung')]]",
            namespaces=NS,
        )
        assert len(line1) >= 1, "result is inaccurate"
        idxs = line1[0].xpath(
            "./page:TextEquiv/@index",
            namespaces=NS,
        )
        assert len(idxs) == len(grps) + 1

    input_file_grp = output_file_grp
    output_file_grp = 'OCR-D-EVAL2'
    with subtests.test(msg="test evaluate (single joined fileGrp)"):
        run_processor(EvaluateLines,
                      input_file_grp=input_file_grp,
                      output_file_grp=output_file_grp,
                      parameter={
                          'match_on': 'index',
                          'confusion': 10,
                          'histogram': True,
                          'metric': 'historic_latin',
                          'gt_level': 2,
                      },
                      **processor_kwargs,
        )
        ws.save_mets()
        assert os.path.isdir(os.path.join(ws.directory, output_file_grp))
        results = list(ws.find_files(file_grp=output_file_grp, mimetype='application/json'))
        assert len(results), "found no output JSON report files"
        assert len(results) == len(pages) + 1
        result0 = next((result for result in results if result.pageId is None), None)
        assert result0 is not None, "found no document-wide JSON report file"
        with open(result0.local_filename) as file:
            result0 = json.load(file)
        for ocr_file_grp, _ in enumerate(grps):
            assert "%d,0" % (ocr_file_grp + 1) in result0
        eval2_result = result0

    for n_ocr, ocr_file_grp in enumerate(grps):
        pair = "%s,%s" % (ocr_file_grp, 'OCR-D-OCR-MULTI')
        pair2 = "%d,0" % (n_ocr + 1)
        assert eval_result[pair]['char-error-rate-mean'] == eval2_result[pair2]['char-error-rate-mean']
        record_property('eval-cer-' + pair, eval_result[pair]['char-error-rate-mean'])

    input_file_grp = 'OCR-D-OCR-PC-ANN'
    output_file_grp = input_file_grp + '-MARK'
    with subtests.test(msg="test mark with hunspell"):
        run_processor(MarkWords,
                      input_file_grp=input_file_grp,
                      output_file_grp=output_file_grp,
                      parameter={
                          'format': 'OOV',
                          'command': 'hunspell -i utf-8 -d de_DE -w',
                          'normalization': {"ſ": "s", "aͤ": "ä", "oͤ": "ö", "uͤ": "ü"},
                      },
                      **processor_kwargs,
        )
        ws.save_mets()
        assert os.path.isdir(os.path.join(ws.directory, output_file_grp))
        results = list(ws.find_files(file_grp=output_file_grp, mimetype=MIMETYPE_PAGE))
        assert len(results), "found no output PAGE files"
        assert len(results) == len(pages)
        result1 = results[0]
        assert os.path.exists(result1.local_filename), "result for first page not found in filesystem"
        tree1 = page_from_file(result1).etree
        ivw1 = tree1.xpath(
            "//page:Word/page:TextEquiv[page:Unicode[contains(text(),'Aufklärung')]]",
            namespaces=NS,
        )
        assert len(ivw1) >= 1, "in-vocabulary word not found"
        oov1 = tree1.xpath(
            "//page:Word/page:TextEquiv[@comments='OOV']",
            namespaces=NS,
        )
        assert len(oov1) >= 1, "out-of-vocabulary word not found"
