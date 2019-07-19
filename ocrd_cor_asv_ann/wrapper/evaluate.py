from __future__ import absolute_import

import os
import math

from ocrd import Processor
from ocrd_utils import getLogger, concat_padded
from ocrd_modelfactory import page_from_file

from .config import OCRD_TOOL
from ..lib.alignment import Alignment

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
        self.alignment = Alignment(logger=LOG)
        
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
        if metric == 'Levenshtein':
            metric = self.alignment.get_levenshtein_distance
        else:
            metric = (lambda ocr, gt, normalization=metric
                      if metric != 'combining-e-umlauts'
                      else None: # NFC / NFKC / historic_latin
                      self.alignment.get_adjusted_distance(
                          ocr, gt, normalization=normalization))
        
        ifgs = self.input_file_grp.split(",") # input file groups
        if len(ifgs) < 2:
            raise Exception("need multiple input file groups to compare")
        
        dists = [0 for _ in ifgs]
        total = [0 for _ in ifgs]
        # get input files:
        ifts = self.zip_input_files(ifgs) # input file tuples
        for ift in ifts:
            file_dists = [0 for _ in ifgs] # sum distances for this file
            file_total = [0 for _ in ifgs] # num characters for this file
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
            for line_id in file_lines[0].keys():
                for i in range(1, len(ift)):
                    if not ift[i]:
                        # file/page was not found in this group
                        continue
                    elif line_id not in file_lines[i]:
                        LOG.error('line "%s" in file %s is missing from input %d / %s',
                                  line_id, ift[i].ID, i, ifgs[i])
                        continue
                    gt_line = file_lines[0][line_id]
                    ocr_line = file_lines[i][line_id]
                    gt_len = len(gt_line)
                    ocr_len = len(ocr_line)
                    if 0.2 * (gt_len + ocr_len) < math.fabs(gt_len - ocr_len) > 5:
                        LOG.warning('line length differs significantly (%d vs %d) for line %s',
                                    gt_len, ocr_len, line_id)
                    dist, chars = metric(ocr_line, gt_line)
                    file_dists[i] += dist
                    file_total[i] += chars
            for i in range(1, len(ift)):
                if not ift[i]:
                    # file/page was not found in this group
                    continue
                LOG.info("CER %s / %s vs %s: %.3f",
                         ift[i].pageId, ifgs[0], ifgs[i], file_dists[i] / file_total[i])
                dists[i] += file_dists[i]
                total[i] += file_total[i]
        for i in range(1, len(ifgs)):
            if not total[i]:
                LOG.warning('%s had no textlines whatsoever', ifgs[i])
                continue
            LOG.info("CER overall / %s vs %s: %.3f",
                     ifgs[0], ifgs[i], dists[i] / total[i])
            
    def zip_input_files(self, ifgs):
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
    
