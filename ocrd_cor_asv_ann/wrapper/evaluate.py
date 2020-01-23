from __future__ import absolute_import

import os
import math

from ocrd import Processor
from ocrd_utils import getLogger, concat_padded
from ocrd_modelfactory import page_from_file

from .config import OCRD_TOOL
from ..lib.alignment import Alignment, Edits

LOG = getLogger('processor.EvaluateLines')
TOOL_NAME = 'ocrd-cor-asv-ann-evaluate'

class EvaluateLines(Processor):
    
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL_NAME]
        kwargs['version'] = OCRD_TOOL['version']
        super(EvaluateLines, self).__init__(*args, **kwargs)
        if not hasattr(self, 'workspace') or not self.workspace:
            # no parameter/workspace for --dump-json or --version (no processing)
            return
        
    def process(self):
        """Align textlines of multiple file groups and calculate distances.
        
        Find files in all input file groups of the workspace for the same
        pageIds (or, as a fallback, the same pageIds at their imageFilename).
        The first file group serves as reference annotation (ground truth).
        
        Open and deserialise PAGE input files, then iterative over the element
        hierarchy down to the TextLine level, looking at each first TextEquiv.
        Align character sequences in all pairs of lines for the same TextLine IDs,
        and calculate the distances using the error metric `metric`. Accumulate
        distances and sequence lengths per file group globally and per file,
        and show each fraction as a CER rate in the log.
        """
        metric = self.parameter['metric']
        confusion = self.parameter['confusion']
        LOG.info('Using evaluation metric "%s".', metric)
        
        ifgs = self.input_file_grp.split(",") # input file groups
        if len(ifgs) < 2:
            raise Exception("need multiple input file groups to compare")
        ifts = self.zip_input_files(ifgs) # input file tuples

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
                file_lines[i] = _page_get_lines(pcgts)
            # compare lines with GT:
            for line_id in file_lines[0].keys():
                for i, input_file in enumerate(ift):
                    if not i:
                        continue
                    elif not input_file:
                        # file/page was not found in this group
                        continue
                    elif line_id not in file_lines[i]:
                        LOG.error('line "%s" in file "%s" is missing from input %d',
                                  line_id, input_file.ID, i)
                        continue
                    gt_line = file_lines[0][line_id]
                    ocr_line = file_lines[i][line_id]
                    gt_len = len(gt_line)
                    ocr_len = len(ocr_line)
                    if 0.2 * (gt_len + ocr_len) < math.fabs(gt_len - ocr_len) > 5:
                        LOG.warning('line "%s" in file "%s" deviates significantly in length (%d vs %d)',
                                    line_id, input_file.ID, gt_len, ocr_len)
                    # align and accumulate edit counts for lines:
                    file_edits[i].add(
                        # not exact (but fast): codepoints
                        self.aligners[i].get_levenshtein_distance(ocr_line, gt_line)
                        if metric == 'Levenshtein' else
                        # exact (but slow): grapheme clusters
                        self.aligners[i].get_adjusted_distance(ocr_line, gt_line,
                                                               # NFC / NFKC / historic_latin
                                                               normalization=metric))
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
                # accumulate edit counts for files
                edits[i].merge(file_edits[i])
        # report overall results
        for i in range(1, len(ifgs)):
            if not edits[i].length:
                LOG.warning('%s had no textlines whatsoever', ifgs[i])
                continue
            LOG.info("%5d lines %.3f±%.3f CER overall / %s vs %s",
                     edits[i].length,
                     edits[i].mean,
                     math.sqrt(edits[i].varia),
                     ifgs[0], ifgs[i])
        if confusion:
            for i in range(1, len(ifgs)):
                LOG.info("most frequent confusion / %s vs %s: %s",
                         ifgs[0], ifgs[i], self.aligners[i].get_confusion(confusion))
            
    def zip_input_files(self, ifgs):
        """Get a list (for each physical page) of tuples (for each input file group) of METS files."""
        ifts = list() # file tuples
        for page_id in self.workspace.mets.physical_pages:
            ifiles = list()
            for ifg in ifgs:
                LOG.debug("adding input file group %s to page %s", ifg, page_id)
                files = self.workspace.mets.find_files(pageId=page_id, fileGrp=ifg)
                if not files:
                    # fall back for missing pageId via Page imageFilename:
                    all_files = self.workspace.mets.find_files(fileGrp=ifg)
                    for file_ in all_files:
                        pcgts = page_from_file(self.workspace.download_file(file_))
                        image_url = pcgts.get_Page().get_imageFilename()
                        img_files = self.workspace.mets.find_files(url=image_url)
                        if img_files and img_files[0].pageId == page_id:
                            files = [file_]
                            break
                if not files:
                    # other fallback options?
                    LOG.error('found no page %s in file group %s',
                              page_id, ifg)
                    ifiles.append(None)
                else:
                    ifiles.append(files[0])
            if ifiles[0]:
                ifts.append(tuple(ifiles))
        return ifts

def _page_get_lines(pcgts):
    '''Get all TextLines in the page.
    
    Iterate the element hierarchy of the page `pcgts` down
    to the TextLine level. For each line, store the element
    ID and its first TextEquiv annotation.
    
    Return the stored dictionary.
    '''
    result = dict()
    regions = pcgts.get_Page().get_TextRegion()
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
    
