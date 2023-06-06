# -*- coding: utf-8
import os
import sys
import logging
import json
import math
import click
from ocrd_models.ocrd_page import parse, parseString
from ocrd_utils import initLogging

from ..lib.alignment import Alignment, Edits
from ..wrapper.evaluate import page_get_lines

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-o', '--output-file', default='-', help="path name of generated report (default: stdout)",
              type=click.Path(dir_okay=False, writable=True, exists=False, allow_dash=True))
@click.option('-n', '--normalization', default='historic_latin', type=click.Choice(
    ["Levenshtein-fast", "Levenshtein", "NFC", "NFKC", "historic_latin"]),
              help='normalize character sequences before alignment/comparison (set Levenshtein for none)')
@click.option('-l', '--gt-level', default=1, type=click.IntRange(1, 3),
              help='GT transcription level to use for historic_latin normlization (1: strongest, 3: none)')
@click.option('-c', '--confusion', default=10, type=click.IntRange(min=0),
              help='show this number of most frequent (non-identity) edits (set 0 for none)')
@click.option('-H', '--histogram', is_flag=True, help='aggregate and compare character histograms')
@click.option('-F', '--file-lists', is_flag=True, help='interpret files as plain text files with one file path per line')
@click.argument('gt_file', type=click.Path(dir_okay=False, exists=True))
@click.argument('ocr_files', type=click.Path(dir_okay=False, exists=True), nargs=-1)
def cli(output_file, normalization, gt_level, confusion, histogram, file_lists, gt_file, ocr_files):
    """Compare text lines by aligning and computing the textual distance and character error rate.
    
    This compares 1:n given PAGE-XML or plain text files.
    
    If `--file-lists` is given and files are plain text,
    then they will be interpreted as (newline-separated)
    lists of path names for single-line text files (for
    Ocropus convention).
    
    Writes a JSON report file to `--output-file`.
    (No error aggregation across files in this CLI.)
    """
    initLogging()
    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.INFO)
    
    caligners = [Alignment(logger=LOG, confusion=bool(confusion)) for _ in ocr_files]
    waligners = [Alignment(logger=LOG) for _ in ocr_files]
    cedits = [Edits(logger=LOG, histogram=bool(histogram)) for _ in ocr_files]
    wedits = [Edits(logger=LOG) for _ in ocr_files]
    LOG.info("processing '%s'", gt_file)
    gt_lines = get_lines(gt_file, file_lists)
    if not gt_lines:
        LOG.critical("file '%s' contains no text lines to compare", gt_file)
        exit(1)
    report = dict()
    for i, ocr_file in enumerate(ocr_files):
        LOG.info("processing '%s'", ocr_file)
        ocr_lines = get_lines(ocr_file, file_lists)
        if not ocr_lines:
            LOG.error("file '%s' contains no text lines to compare", ocr_file)
            continue
        pair = gt_file + ',' + ocr_file
        if isinstance(ocr_lines, dict):
            # from PAGE-XML file
            line_ids = ocr_lines.keys()
        else:
            # from plain text file
            line_ids = range(len(ocr_lines))
        for line_id in line_ids:
            lines = report.setdefault(pair, dict()).setdefault('lines', list())
            if isinstance(gt_lines, dict):
                has_line = line_id in gt_lines
            else:
                has_line = line_id < len(gt_lines)
            if not has_line:
                LOG.error("line '%s' in file '%s' is missing from GT file '%s'",
                          str(line_id), ocr_file, gt_file)
                lines.append({line_id: 'missing'})
                continue
            gt_line = gt_lines[line_id].strip()
            gt_len = len(gt_line)
            gt_words = gt_line.split()
            ocr_line = ocr_lines[line_id].strip()
            ocr_len = len(ocr_line)
            ocr_words = ocr_line.split()
            if 0.2 * (gt_len + ocr_len) < math.fabs(gt_len - ocr_len) > 5:
                LOG.warning('line "%s" in file "%s" deviates significantly in length (%d vs %d)',
                            str(line_id), ocr_file, gt_len, ocr_len)
            if normalization == 'Levenshtein-fast':
                # not exact (but fast): codepoints
                cdist = caligners[i].get_levenshtein_distance(ocr_line, gt_line)
                wdist = waligners[i].get_levenshtein_distance(ocr_words, gt_words)
            else:
                # exact (but slow): grapheme clusters
                cdist = caligners[i].get_adjusted_distance(ocr_line, gt_line,
                                                           # Levenshtein / NFC / NFKC / historic_latin
                                                           normalization=normalization,
                                                           gtlevel=gt_level)
                wdist = waligners[i].get_adjusted_distance(ocr_words, gt_words,
                                                           # Levenshtein / NFC / NFKC / historic_latin
                                                           normalization=normalization,
                                                           gtlevel=gt_level)
            _, conf = Alignment.best_alignment(ocr_line, gt_line, True)
            cedits[i].add(cdist, ocr_line, gt_line)
            wedits[i].add(wdist, ocr_words, gt_words)
            lines.append({line_id: {
                'char-length': gt_len,
                'char-error-rate': cdist,
                'word-error-rate': wdist,
                'gt': gt_line,
                'ocr': ocr_line,
                'edits': repr(conf)}})
        # report results
        LOG.info("%5d lines %.3f±%.3f CER %.3f±%.3f WER %s vs %s",
                 cedits[i].length,
                 cedits[i].mean, math.sqrt(cedits[i].varia),
                 wedits[i].mean, math.sqrt(wedits[i].varia),
                 ocr_file, gt_file)
        report[pair]['num-lines'] = cedits[i].length
        report[pair]['char-error-rate-mean'] = cedits[i].mean
        report[pair]['char-error-rate-varia'] = cedits[i].varia
        report[pair]['word-error-rate-mean'] = wedits[i].mean
        report[pair]['word-error-rate-varia'] = wedits[i].varia
        if confusion:
            if not cedits[i].length:
                continue
            conf = caligners[i].get_confusion(confusion)
            LOG.info("most frequent confusion / %s vs %s: %s",
                     gt_file, ocr_file, conf)
            report[pair]['confusion'] = repr(conf)
        if histogram:
            hist = cedits[i].hist()
            LOG.info("character histograms / %s vs %s: %s",
                     gt_file, ocr_file, hist)
            report[pair]['histogram'] = repr(hist)
    if output_file == '-':
        output = sys.stdout
    else:
        output = open(output_file, 'w')
    json.dump(report, output, indent=2, ensure_ascii=False)

def get_lines(fname, flist=False):
    with open(fname, 'r') as fd:
        rawlines = [line.rstrip('\r\n') for line in fd.readlines()]
    try:
        # PAGE-XML case
        if rawlines and rawlines[0].startswith('<?xml'):
            rawlines[0] = rawlines[0][rawlines[0].index('?>')+2:]
        pcgts = parseString(''.join(rawlines))
        #pcgts = parse(fname)
        lines = page_get_lines(pcgts)
    except Exception:
        # plaintext case
        lines = rawlines
        if flist:
            # ocropy style (e.g. -F <(ls -1 *.gt.txt) <(ls -1 *.ocr.txt))
            files = lines
            lines = dict()
            for fname in files:
                with open(fname, 'r') as fd:
                    if fname.endswith('.txt'):
                        dirname, basename = os.path.split(fname)
                        parts = basename.split('.')
                        fname = os.path.join(dirname, parts[0])
                    lines[fname] = fd.readline()
    return lines
