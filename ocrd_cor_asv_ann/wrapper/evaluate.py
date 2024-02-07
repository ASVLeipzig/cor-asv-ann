from __future__ import absolute_import

import os
import math
import json
import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd import Processor
from ocrd_utils import (
    getLogger,
    assert_file_grp_cardinality,
    make_file_id,
    MIMETYPE_PAGE
)
from ocrd_modelfactory import page_from_file

from .config import OCRD_TOOL
from ..lib.alignment import Alignment, Edits, splitwords

TOOL_NAME = 'ocrd-cor-asv-ann-evaluate'

@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann_evaluate(*args, **kwargs):
    return ocrd_cli_wrap_processor(EvaluateLines, *args, **kwargs)

class EvaluateLines(Processor):
    
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL_NAME]
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)
        
    def process(self):
        """Align textlines of alternative annotations and calculate distances for error rates.

        \b
        Alternative input annotations derive from either:
        - multiple ``TextEquiv/@index`` within one file group and file, or
        - multiple file groups (i.e. multiple files for the same page ID, with
          the same TextLine id or coords, using only the first TextEquiv).

        The first annotation (i.e. index or file group) serves as reference
        (ground truth).
        
        Open and deserialise PAGE input files, then iterate over the element
        hierarchy down to the TextLine level.

        \b
        Now, align character sequences in all pairs of lines for the same 
        - TextLine IDs across files (if `match_on=id`) or 
        - TextLine boundary points across files (if `match_on=coords`) or 
        - TextLine baseline points across files (if `match_on=baseline`) or
        - ``TextLine/TextEquiv/@index`` (if `match_on=index`),
        and calculate the distances using the error metric `metric`. Accumulate
        distances and sequence lengths per file group globally and per file,
        and show each fraction as a CER and WER rate in the output JSON reports.
        """
        assert_file_grp_cardinality(self.output_file_grp, 1)

        LOG = getLogger('processor.EvaluateLines')

        metric = self.parameter['metric']
        gtlevel = self.parameter['gt_level']
        confusion = self.parameter['confusion']
        histogram = self.parameter['histogram']
        LOG.info('Using evaluation metric "%s".', metric)
        if metric == 'Levenshtein-fast' and confusion > 0:
            LOG.warning('There will be no confusion statistics with this metric.')
        match = self.parameter['match_on']
        
        ifgs = self.input_file_grp.split(",") # input file groups
        if match == 'index':
            assert_file_grp_cardinality(self.input_file_grp, 1,
                                        msg="only 1 input fileGrp when match_on=index")
        elif len(ifgs) < 2:
            raise Exception("need multiple input file groups to compare when match_on!=index")
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE) # input file tuples

        # get separate aligners (1 more than needed), because
        # they are stateful (confusion counts):
        caligners = [Alignment(logger=LOG, confusion=bool(confusion)) for _ in ifgs]
        waligners = [Alignment(logger=LOG) for _ in ifgs]
        
        # running edit counts/mean/variance for each file group:
        cedits = [Edits(logger=LOG, histogram=histogram) for _ in ifgs]
        wedits = [Edits(logger=LOG) for _ in ifgs]
        # get input files:
        for ift in ifts:
            # running edit counts/mean/variance for each file group for this file:
            file_cedits = [Edits(logger=LOG, histogram=histogram) for _ in ifgs]
            file_wedits = [Edits(logger=LOG) for _ in ifgs]
            # get input lines:
            if match == 'index':
                input_file = ift[0]
                LOG.info("processing page %s", input_file.pageId)
                LOG.info("INPUT FILE: %s", input_file.ID)
                pcgts = page_from_file(self.workspace.download_file(input_file))
                line_indexes = page_get_lines(pcgts, match)
                line_ids = line_indexes.keys()
                file_lines = []
                for line_id in line_ids:
                    indexes = line_indexes[line_id].keys()
                    if not indexes:
                        continue
                    for _ in range(len(file_lines), max(indexes) + 1):
                        file_lines.append(dict())
                    for _ in range(len(cedits), max(indexes) + 1):
                        cedits.append(Edits(logger=LOG, histogram=histogram))
                        wedits.append(Edits(logger=LOG))
                        caligners.append(Alignment(logger=LOG, confusion=bool(confusion)))
                        waligners.append(Alignment(logger=LOG))
                    for _ in range(len(file_cedits), max(indexes) + 1):
                        file_cedits.append(Edits(logger=LOG, histogram=histogram))
                        file_wedits.append(Edits(logger=LOG))
                    for index in indexes:
                        file_lines[index][line_id] = line_indexes[line_id][index]
            else:
                file_lines = [{} for _ in ifgs] # line dicts for this file
                for i, input_file in enumerate(ift):
                    if not i:
                        LOG.info("processing page %s", input_file.pageId)
                    if not input_file:
                        # file/page was not found in this group
                        continue
                    LOG.info("INPUT FILE for %s: %s", ifgs[i], input_file.ID)
                    pcgts = page_from_file(self.workspace.download_file(input_file))
                    file_lines[i] = page_get_lines(pcgts, match)
            # compare lines with GT:
            report = dict()
            gt_lines = file_lines[0]
            for line_id in gt_lines.keys():
                for i, input_lines in enumerate(file_lines):
                    if not i:
                        continue
                    if match == 'index':
                        pair = '%d,0' % i
                    else:
                        pair = ifgs[i] + ',' + ifgs[0]
                        input_file = ift[i]
                    lines = report.setdefault(pair, dict()).setdefault('lines', list())
                    if not input_lines or not input_file:
                        # file/page was not found in this group
                        continue
                    elif line_id not in input_lines:
                        LOG.error('line "%s" in file "%s" is missing from input %d',
                                  line_id, input_file.ID, i)
                        lines.append({line_id: 'missing'})
                        continue
                    gt_line = gt_lines[line_id]
                    gt_len = len(gt_line)
                    gt_words = splitwords(gt_line)
                    ocr_line = input_lines[line_id]
                    ocr_len = len(ocr_line)
                    ocr_words = splitwords(ocr_line)
                    if 0.2 * (gt_len + ocr_len) < math.fabs(gt_len - ocr_len) > 5:
                        LOG.warning('line "%s" in file "%s" deviates significantly in length (%d vs %d)',
                                    line_id, input_file.ID, gt_len, ocr_len)
                    if metric == 'Levenshtein-fast':
                        # not exact (but fast): codepoints
                        cdist, clen = caligners[i].get_levenshtein_distance(ocr_line, gt_line)
                        wdist, wlen = waligners[i].get_levenshtein_distance(ocr_words, gt_words)
                    else:
                        # exact (but slow): grapheme clusters
                        cdist, clen = caligners[i].get_adjusted_distance(
                            ocr_line, gt_line,
                            # Levenshtein / NFC / NFKC / historic_latin
                            normalization=metric,
                            gtlevel=gtlevel)
                        wdist, wlen = waligners[i].get_adjusted_distance(
                            ocr_words, gt_words,
                            # Levenshtein / NFC / NFKC / historic_latin
                            normalization=metric,
                            gtlevel=gtlevel)
                    # align and accumulate edit counts for lines:
                    file_cedits[i].add(cdist, clen, ocr_line, gt_line)
                    file_wedits[i].add(wdist, wlen, ocr_words, gt_words)
                    # todo: maybe it could be useful to retrieve and store the alignments, too
                    lines.append({line_id: {
                        'char-length': gt_len,
                        'char-error-rate': cdist / clen if clen else 0,
                        'word-error-rate': wdist / wlen if wlen else 0,
                        'gt': gt_line,
                        'ocr': ocr_line}})
            
            # report results for file
            for i, input_lines in enumerate(file_lines):
                if not i:
                    continue
                elif not input_lines:
                    # file/page was not found in this group
                    continue
                if match == 'index':
                    pair = '%d,0' % i
                else:
                    pair = ifgs[i] + ',' + ifgs[0]
                    input_file = ift[i]
                LOG.info("%5d lines %.3f±%.3f CER %.3f±%.3f WER %s / %s",
                         file_cedits[i].steps,
                         file_cedits[i].mean, math.sqrt(file_cedits[i].varia),
                         file_wedits[i].mean, math.sqrt(file_wedits[i].varia),
                         input_file.pageId, pair)
                report[pair] = {}
                report[pair]['num-lines'] = file_cedits[i].steps
                report[pair]['num-words'] = file_wedits[i].length
                report[pair]['num-chars'] = file_cedits[i].length
                report[pair]['char-error-rate-mean'] = file_cedits[i].mean
                report[pair]['char-error-rate-varia'] = file_cedits[i].varia
                report[pair]['word-error-rate-mean'] = file_wedits[i].mean
                report[pair]['word-error-rate-varia'] = file_wedits[i].varia
                # accumulate edit counts for files
                cedits[i].merge(file_cedits[i])
                wedits[i].merge(file_wedits[i])
            
            # write back result to page report
            file_id = make_file_id(ift[0], self.output_file_grp)
            file_path = os.path.join(self.output_file_grp, file_id + '.json')
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype='application/json',
                content=json.dumps(report, indent=2, ensure_ascii=False))
            
        # report overall results
        report = dict()
        for i in range(len(caligners)):
            if not i:
                continue
            if match == 'index':
                src = 'index %d' % i
                pair = '%d,0' % i
            else:
                src = ifgs[i]
                pair = ifgs[i] + ',' + ifgs[0]
            if not cedits[i].steps:
                LOG.warning('%s had no textlines whatsoever', src)
                continue
            LOG.info("%5d lines %.3f±%.3f CER %.3f±%.3f WER overall / %s",
                     cedits[i].steps,
                     cedits[i].mean, math.sqrt(cedits[i].varia),
                     wedits[i].mean, math.sqrt(wedits[i].varia),
                     pair)
            report[pair] = {
                'num-lines': cedits[i].steps,
                'num-words': wedits[i].length,
                'num-chars': cedits[i].length,
                'char-error-rate-mean': cedits[i].mean,
                'char-error-rate-varia': cedits[i].varia,
                'word-error-rate-mean': wedits[i].mean,
                'word-error-rate-varia': wedits[i].varia,
            }
            if confusion:
                conf = caligners[i].get_confusion(confusion)
                LOG.info("most frequent confusion / %s: %s", pair, conf)
                report[pair]['confusion'] = repr(conf)
            if histogram:
                hist = cedits[i].hist()
                LOG.info("character histograms / %s: %s", pair, hist)
                report[pair]['histogram'] = repr(hist)
        # write back result to overall report
        file_id = self.output_file_grp
        file_path = os.path.join(self.output_file_grp, file_id + '.json')
        self.workspace.add_file(
            ID=file_id,
            file_grp=self.output_file_grp,
            pageId=None,
            local_filename=file_path,
            mimetype='application/json',
            content=json.dumps(report, indent=2, ensure_ascii=False))

def _linekey(line, match_on):
    if match_on == 'id':
        return line.id
    if match_on == 'baseline':
        if line.Baseline is None:
            LOG.error("cannot extract baseline from line '%s'", line.id)
            return line.Coords.points
        return line.Baseline.points
    if match_on == 'coords':
        return line.Coords.points

def page_get_lines(pcgts, match_on):
    '''Get all TextLines in the page.
    
    Iterate the element hierarchy of the page `pcgts` down
    to the TextLine level. For each line, store the element
    ID and its first TextEquiv annotation.
    
    Return the stored dictionary.
    '''
    LOG = getLogger('processor.EvaluateLines')
    result = dict()
    regions = pcgts.get_Page().get_AllRegions(classes=['Text'], order='reading-order')
    if not regions:
        LOG.warning("Page contains no text regions")
    for region in regions:
        lines = region.get_TextLine()
        if not lines:
            LOG.warning("Region '%s' contains no text lines", region.id)
            continue
        for line in lines:
            textequivs = line.get_TextEquiv()
            if not textequivs:
                LOG.warning("Line '%s' contains no text results", line.id)
                continue
            if match_on == 'index':
                for i, textequiv in enumerate(textequivs):
                    index = textequiv.index or i
                    lined = result.setdefault(line.id, dict())
                    if index in lined:
                        LOG.warning("Line '%s' contains TextEquiv with and without @index", line.id)
                    lined[index] = textequiv.Unicode
            else:
                result[_linekey(line, match_on)] = textequivs[0].Unicode
    return result
    
