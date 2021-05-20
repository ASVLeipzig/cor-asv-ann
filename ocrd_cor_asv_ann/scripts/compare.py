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
@click.option('-l', '--gt-level', default=1, type=click.IntRange(1,3),
              help='GT transcription level to use for historic_latin normlization (1: strongest, 3: none)')
@click.option('-c', '--confusion', default=10, type=click.IntRange(min=0),
              help='show this number of most frequent (non-identity) edits (set 0 for none)')
@click.option('-F', '--file-lists', is_flag=True, help='interpret files as plain text files with one file path per line')
@click.argument('gt_file', type=click.Path(dir_okay=False, exists=True))
@click.argument('ocr_files', type=click.Path(dir_okay=False, exists=True), nargs=-1)
def cli(output_file, normalization, gt_level, confusion, file_lists, gt_file, ocr_files):
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
    
    aligners = [Alignment(logger=LOG, confusion=bool(confusion)) for _ in ocr_files]
    edits = [Edits(logger=LOG) for _ in ocr_files]
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
            gt_line = gt_lines[line_id]
            ocr_line = ocr_lines[line_id]
            gt_len = len(gt_line)
            ocr_len = len(ocr_line)
            if 0.2 * (gt_len + ocr_len) < math.fabs(gt_len - ocr_len) > 5:
                LOG.warning('line "%s" in file "%s" deviates significantly in length (%d vs %d)',
                            str(line_id), ocr_file, gt_len, ocr_len)
            if normalization == 'Levenshtein-fast':
                # not exact (but fast): codepoints
                dist = aligners[i].get_levenshtein_distance(ocr_line, gt_line)
            else:
                # exact (but slow): grapheme clusters
                dist = aligners[i].get_adjusted_distance(ocr_line, gt_line,
                                                         # Levenshtein / NFC / NFKC / historic_latin
                                                         normalization=normalization)
            edits[i].add(dist)
            lines.append({line_id: {'length': gt_len, 'distance': dist}})
        # report results
        LOG.info("%5d lines %.3f±%.3f CER %s vs %s",
                 edits[i].length, edits[i].mean,
                 math.sqrt(edits[i].varia),
                 ocr_file, gt_file)
        report[pair]['length'] = edits[i].length
        report[pair]['distance-mean'] = edits[i].mean
        report[pair]['distance-varia'] = edits[i].varia
        if confusion:
            if not edits[i].length:
                continue
            conf = aligners[i].get_confusion(confusion)
            LOG.info("most frequent confusion / %s vs %s: %s",
                     gt_file, ocr_file, conf)
            report[pair]['confusion'] = repr(conf)
    if output_file == '-':
        output = sys.stdout
    else:
        output = open(output_file, 'w')
    json.dump(report, output, indent=2)

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
            lines = []
            for fname in files:
                with open(fname, 'r') as fd:
                    lines.append(fd.readline())
    return lines
