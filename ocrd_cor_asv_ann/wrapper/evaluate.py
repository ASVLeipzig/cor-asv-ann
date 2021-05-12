from __future__ import absolute_import

import os
import math
import json

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
        confusion = self.parameter['confusion']
        LOG.info('Using evaluation metric "%s".', metric)
        
        ifgs = self.input_file_grp.split(",") # input file groups
        if len(ifgs) < 2:
            raise Exception("need multiple input file groups to compare")
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE) # input file tuples

        # get separate aligners (1 more than needed), because
        # they are stateful (confusion counts):
        self.aligners = [Alignment(logger=LOG, confusion=bool(confusion)) for _ in ifgs]
        
        # running edit counts/mean/variance for each file group:
        edits = [Edits(logger=LOG) for _ in ifgs]
        # get input files:
        for ift in ifts:
            # running edit counts/mean/variance for each file group for this file:
            file_edits = [Edits(logger=LOG) for _ in ifgs]
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
                    ocr_line = file_lines[i][line_id]
                    gt_len = len(gt_line)
                    ocr_len = len(ocr_line)
                    if 0.2 * (gt_len + ocr_len) < math.fabs(gt_len - ocr_len) > 5:
                        LOG.warning('line "%s" in file "%s" deviates significantly in length (%d vs %d)',
                                    line_id, input_file.ID, gt_len, ocr_len)
                    if metric == 'Levenshtein-fast':
                        # not exact (but fast): codepoints
                        dist = self.aligners[i].get_levenshtein_distance(ocr_line, gt_line)
                    else:
                        # exact (but slow): grapheme clusters
                        dist = self.aligners[i].get_adjusted_distance(ocr_line, gt_line,
                                                                      # Levenshtein / NFC / NFKC / historic_latin
                                                                      normalization=metric)
                    # align and accumulate edit counts for lines:
                    file_edits[i].add(dist)
                    # todo: maybe it could be useful to retrieve and store the alignments, too
                    lines.append({line_id: {'length': gt_len, 'distance': dist}})
            
            # report results for file
            for i, input_file in enumerate(ift):
                if not i:
                    continue
                elif not input_file:
                    # file/page was not found in this group
                    continue
                LOG.info("%5d lines %.3f±%.3f CER %s / %s vs %s",
                         file_edits[i].length,
                         file_edits[i].mean,
                         math.sqrt(file_edits[i].varia),
                         input_file.pageId, ifgs[0], ifgs[i])
                pair = ifgs[0] + ',' + ifgs[i]
                report[pair]['length'] = file_edits[i].length
                report[pair]['distance-mean'] = file_edits[i].mean
                report[pair]['distance-varia'] = file_edits[i].varia
                # accumulate edit counts for files
                edits[i].merge(file_edits[i])
            
            # write back result to page report
            file_id = make_file_id(ift[0], self.output_file_grp)
            file_path = os.path.join(self.output_file_grp, file_id + '.json')
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype='application/json',
                content=json.dumps(report, indent=2))
            
        # report overall results
        report = dict()
        for i in range(1, len(ifgs)):
            if not edits[i].length:
                LOG.warning('%s had no textlines whatsoever', ifgs[i])
                continue
            LOG.info("%5d lines %.3f±%.3f CER overall / %s vs %s",
                     edits[i].length,
                     edits[i].mean,
                     math.sqrt(edits[i].varia),
                     ifgs[0], ifgs[i])
            report[ifgs[0] + ',' + ifgs[i]] = {
                'length': edits[i].length,
                'distance-mean': edits[i].mean,
                'distance-varia': edits[i].varia
            }
        if confusion:
            for i in range(1, len(ifgs)):
                if not edits[i].length:
                    continue
                conf = self.aligners[i].get_confusion(confusion)
                LOG.info("most frequent confusion / %s vs %s: %s",
                         ifgs[0], ifgs[i], conf)
                report[ifgs[0] + ',' + ifgs[i]]['confusion'] = repr(conf)
        # write back result to overall report
        file_id = self.output_file_grp
        file_path = os.path.join(self.output_file_grp, file_id + '.json')
        self.workspace.add_file(
            ID=file_id,
            file_grp=self.output_file_grp,
            pageId=None,
            local_filename=file_path,
            mimetype='application/json',
            content=json.dumps(report, indent=2))

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
    
