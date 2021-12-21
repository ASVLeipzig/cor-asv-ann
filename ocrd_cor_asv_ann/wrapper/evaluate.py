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
from ..lib.alignment import Alignment, Edits

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
        """Align textlines of multiple file groups and calculate distances.
        
        Find files in all input file groups of the workspace for the same
        pageIds. The first file group serves as reference annotation (ground truth).
        
        Open and deserialise PAGE input files, then iterate over the element
        hierarchy down to the TextLine level, looking at each first TextEquiv.
        Align character sequences in all pairs of lines for the same TextLine IDs,
        and calculate the distances using the error metric `metric`. Accumulate
        distances and sequence lengths per file group globally and per file,
        and show each fraction as a CER rate in the log.
        """
        assert_file_grp_cardinality(self.output_file_grp, 1)

        LOG = getLogger('processor.EvaluateLines')

        metric = self.parameter['metric']
        gtlevel = self.parameter['gt_level']
        confusion = self.parameter['confusion']
        histogram = self.parameter['histogram']
        LOG.info('Using evaluation metric "%s".', metric)
        
        ifgs = self.input_file_grp.split(",") # input file groups
        if len(ifgs) < 2:
            raise Exception("need multiple input file groups to compare")
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE) # input file tuples

        # get separate aligners (1 more than needed), because
        # they are stateful (confusion counts):
        self.caligners = [Alignment(logger=LOG, confusion=bool(confusion)) for _ in ifgs]
        self.waligners = [Alignment(logger=LOG) for _ in ifgs]
        
        # running edit counts/mean/variance for each file group:
        cedits = [Edits(logger=LOG, histogram=histogram) for _ in ifgs]
        wedits = [Edits(logger=LOG) for _ in ifgs]
        # get input files:
        for ift in ifts:
            # running edit counts/mean/variance for each file group for this file:
            file_cedits = [Edits(logger=LOG, histogram=histogram) for _ in ifgs]
            file_wedits = [Edits(logger=LOG) for _ in ifgs]
            # get input lines:
            file_lines = [{} for _ in ifgs] # line dicts for this file
            for i, input_file in enumerate(ift):
                if not i:
                    LOG.info("processing page %s", input_file.pageId)
                if not input_file:
                    # file/page was not found in this group
                    continue
                LOG.info("INPUT FILE for %s: %s", ifgs[i], input_file.ID)
                pcgts = page_from_file(self.workspace.download_file(input_file))
                file_lines[i] = page_get_lines(pcgts)
            # compare lines with GT:
            report = dict()
            for line_id in file_lines[0].keys():
                for i, input_file in enumerate(ift):
                    if not i:
                        continue
                    pair = ifgs[0] + ',' + ifgs[i]
                    lines = report.setdefault(pair, dict()).setdefault('lines', list())
                    if not input_file:
                        # file/page was not found in this group
                        continue
                    elif line_id not in file_lines[i]:
                        LOG.error('line "%s" in file "%s" is missing from input %d',
                                  line_id, input_file.ID, i)
                        lines.append({line_id: 'missing'})
                        continue
                    gt_line = file_lines[0][line_id]
                    gt_len = len(gt_line)
                    gt_words = gt_line.split()
                    ocr_line = file_lines[i][line_id]
                    ocr_len = len(ocr_line)
                    ocr_words = ocr_line.split()
                    if 0.2 * (gt_len + ocr_len) < math.fabs(gt_len - ocr_len) > 5:
                        LOG.warning('line "%s" in file "%s" deviates significantly in length (%d vs %d)',
                                    line_id, input_file.ID, gt_len, ocr_len)
                    if metric == 'Levenshtein-fast':
                        # not exact (but fast): codepoints
                        cdist = self.caligners[i].get_levenshtein_distance(ocr_line, gt_line)
                        wdist = self.waligners[i].get_levenshtein_distance(ocr_words, gt_words)
                    else:
                        # exact (but slow): grapheme clusters
                        cdist = self.caligners[i].get_adjusted_distance(ocr_line, gt_line,
                                                                        # Levenshtein / NFC / NFKC / historic_latin
                                                                        normalization=metric,
                                                                        gtlevel=gtlevel)
                        wdist = self.waligners[i].get_adjusted_distance(ocr_words, gt_words,
                                                                        # Levenshtein / NFC / NFKC / historic_latin
                                                                        normalization=metric,
                                                                        gtlevel=gtlevel)
                    # align and accumulate edit counts for lines:
                    file_cedits[i].add(cdist, ocr_line, gt_line)
                    file_wedits[i].add(wdist, ocr_words, gt_words)
                    # todo: maybe it could be useful to retrieve and store the alignments, too
                    lines.append({line_id: {
                        'char-length': gt_len,
                        'char-error-rate': cdist,
                        'word-error-rate': wdist,
                        'gt': gt_line,
                        'ocr': ocr_line}})
            
            # report results for file
            for i, input_file in enumerate(ift):
                if not i:
                    continue
                elif not input_file:
                    # file/page was not found in this group
                    continue
                LOG.info("%5d lines %.3f±%.3f CER %.3f±%.3f WER %s / %s vs %s",
                         file_cedits[i].length,
                         file_cedits[i].mean, math.sqrt(file_cedits[i].varia),
                         file_wedits[i].mean, math.sqrt(file_wedits[i].varia),
                         input_file.pageId, ifgs[0], ifgs[i])
                pair = ifgs[0] + ',' + ifgs[i]
                report[pair]['num-lines'] = file_cedits[i].length
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
        for i in range(1, len(ifgs)):
            if not cedits[i].length:
                LOG.warning('%s had no textlines whatsoever', ifgs[i])
                continue
            LOG.info("%5d lines %.3f±%.3f CER %.3f±%.3f WER overall / %s vs %s",
                     cedits[i].length,
                     cedits[i].mean, math.sqrt(cedits[i].varia),
                     wedits[i].mean, math.sqrt(wedits[i].varia),
                     ifgs[0], ifgs[i])
            report[ifgs[0] + ',' + ifgs[i]] = {
                'num-lines': cedits[i].length,
                'char-error-rate-mean': cedits[i].mean,
                'char-error-rate-varia': cedits[i].varia,
                'word-error-rate-mean': wedits[i].mean,
                'word-error-rate-varia': wedits[i].varia,
            }
        if confusion:
            for i in range(1, len(ifgs)):
                if not cedits[i].length:
                    continue
                conf = self.caligners[i].get_confusion(confusion)
                LOG.info("most frequent confusion / %s vs %s: %s",
                         ifgs[0], ifgs[i], conf)
                report[ifgs[0] + ',' + ifgs[i]]['confusion'] = repr(conf)
        if histogram:
            for i in range(1, len(ifgs)):
                if not cedits[i].length:
                    continue
                hist = cedits[i].hist()
                LOG.info("character histograms / %s vs %s: %s",
                         ifgs[0], ifgs[i], hist)
                report[ifgs[0] + ',' + ifgs[i]]['histogram'] = repr(hist)
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

def page_get_lines(pcgts):
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
            result[line.id] = textequivs[0].Unicode
    return result
    
