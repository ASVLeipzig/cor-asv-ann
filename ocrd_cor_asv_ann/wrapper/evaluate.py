from __future__ import absolute_import

import os
import math
import json
from typing import Optional, get_args

import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd import Processor, Workspace, OcrdPageResult
from ocrd_models import OcrdPage, OcrdFileType
from ocrd_modelfactory import page_from_file
from ocrd_utils import (
    config,
    make_file_id,
    MIMETYPE_PAGE
)

from ..lib.alignment import Alignment, Edits, splitwords


@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann_evaluate(*args, **kwargs):
    return ocrd_cli_wrap_processor(EvaluateLines, *args, **kwargs)

class EvaluateLines(Processor):
    max_workers = 1 # no forking (would require syncing global aligners and edits across pages)

    @property
    def executable(self):
        return 'ocrd-cor-asv-ann-evaluate'

    @property
    def metadata_filename(self) -> str:
        return os.path.join('wrapper', 'ocrd-tool.json')

    def zip_input_files(self, **kwargs):
        kwargs['mimetype'] = MIMETYPE_PAGE
        kwargs['require_first'] = True
        return super().zip_input_files(**kwargs)

    def setup(self):
        self.logger.info('Using evaluation metric "%s"', self.parameter['metric'])
        if self.parameter['metric'] == 'Levenshtein-fast' and self.parameter['confusion'] > 0:
            self.logger.warning('There will be no confusion statistics with this metric.')

    def verify(self):
        if not super().verify():
            return False
        # cannot be formulated in ocrd-tool.json:
        input_file_grp_cardinality = len(self.input_file_grps)
        if self.parameter['match_on'] == 'index':
            assert input_file_grp_cardinality == 1, \
                "only 1 input fileGrp when match_on==index"
        else:
            assert input_file_grp_cardinality > 1, \
                "need multiple input file groups to compare when match_on!=index"
        return True

    def input_pair(self, i: int):
        assert i > 0
        if self.parameter['match_on'] == 'index':
            pair = '%d,0' % i
        else:
            pair = self.input_file_grps[i] + ',' + self.input_file_grps[0]
        return pair

    def input_name(self, i: int):
        assert i > 0
        if self.parameter['match_on'] == 'index':
            name = 'index %d' % i
        else:
            name = self.input_file_grps[i]
        return name

    def report_pair(self, report: dict, i: int, cedits:list=None, wedits:list=None) -> None:
        if cedits is None:
            cedits = self.cedits
        if wedits is None:
            wedits = self.wedits
        pair = self.input_pair(i)
        report[pair][''] = self.input_name(i)
        report[pair]['num-lines'] = cedits[i].steps
        report[pair]['num-words'] = wedits[i].length
        report[pair]['num-chars'] = cedits[i].length
        report[pair]['char-error-rate-mean'] = cedits[i].mean
        report[pair]['char-error-rate-varia'] = cedits[i].varia
        report[pair]['word-error-rate-mean'] = wedits[i].mean
        report[pair]['word-error-rate-varia'] = wedits[i].varia
        report[pair]['char-error-worst-lines'] = [str(example) for example in cedits[i].worst]
        #report[pair]['word-error-worst-lines'] = [str(example) for example in cedits[i].worst]

    def process_workspace(self, workspace: Workspace) -> None:
        self.input_file_grps = self.input_file_grp.split(',')

        # get separate aligners (1 more than needed), because
        # they are stateful (confusion counts):
        self.caligners = [Alignment(logger=self.logger, confusion=bool(self.parameter['confusion']))
                          for _ in self.input_file_grps]
        self.waligners = [Alignment(logger=self.logger)
                          for _ in self.input_file_grps]

        # running edit counts/mean/variance for each file group:
        self.cedits = [Edits(logger=self.logger, histogram=self.parameter['histogram'])
                       for _ in self.input_file_grps]
        self.wedits = [Edits(logger=self.logger)
                       for _ in self.input_file_grps]
        super().process_workspace(workspace)

        # report overall results
        report = dict()
        for i in range(len(self.caligners)):
            if i == 0:
                # Ground Truth input
                continue
            pair = self.input_pair(i)
            if not self.cedits[i].steps:
                self.logger.warning('%s had no textlines whatsoever', self.input_name(i))
                continue
            self.logger.info("%5d lines %.3f±%.3f CER %.3f±%.3f WER overall / %s",
                             self.cedits[i].steps,
                             self.cedits[i].mean, math.sqrt(self.cedits[i].varia),
                             self.wedits[i].mean, math.sqrt(self.wedits[i].varia),
                             pair)
            report[pair] = dict()
            self.report_pair(report, i, self.cedits, self.wedits)
            if self.parameter['confusion']:
                conf = self.caligners[i].get_confusion(self.parameter['confusion'])
                self.logger.info("most frequent confusion / %s: %s", pair, conf)
                report[pair]['confusion'] = repr(conf)
            if self.parameter['histogram']:
                hist = self.cedits[i].hist()
                self.logger.info("character histograms / %s: %s", pair, hist)
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
        del self.caligners
        del self.waligners
        del self.cedits
        del self.wedits

    def process_page_file(self, *input_files : Optional[OcrdFileType]) -> None:
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
        # from core's ocrd.processor.base
        input_pcgts : List[Optional[OcrdPage]] = [None] * len(input_files)
        page_id = input_files[0].pageId
        self._base_logger.info("processing page %s", page_id)
        for i, input_file in enumerate(input_files):
            assert isinstance(input_file, get_args(OcrdFileType))
            self._base_logger.debug(f"parsing file {input_file.ID} for page {page_id}")
            try:
                page_ = page_from_file(input_file)
                assert isinstance(page_, OcrdPage)
                input_pcgts[i] = page_
            except ValueError as err:
                # not PAGE and not an image to generate PAGE for
                self._base_logger.error(f"non-PAGE input for page {page_id}: {err}")
        output_file_id = make_file_id(input_files[0], self.output_file_grp)
        output_file = next(self.workspace.mets.find_files(ID=output_file_id), None)
        if output_file and config.OCRD_EXISTING_OUTPUT != 'OVERWRITE':
            # short-cut avoiding useless computation:
            raise FileExistsError(
                f"A file with ID=={output_file_id} already exists {output_file} and neither force nor ignore are set"
            )
        # our own version of it
        metric = self.parameter['metric']
        gtlevel = self.parameter['gt_level']
        confusion = self.parameter['confusion']
        histogram = self.parameter['histogram']
        match = self.parameter['match_on']
        
        # running edit counts/mean/variance for each file group for this file:
        file_cedits = [Edits(logger=self.logger, histogram=histogram)
                       for _ in self.input_file_grps]
        file_wedits = [Edits(logger=self.logger)
                       for _ in self.input_file_grps]
        # get input lines:
        if match == 'index':
            pcgts = input_pcgts[0]
            file_lines = []
            line_indexes = page_get_lines(pcgts, match, logger=self.logger)
            for line_id in line_indexes:
                for index in line_indexes[line_id]:
                    if len(file_lines) <= index:
                        file_lines.append(dict())
                    file_lines[index][line_id] = line_indexes[line_id][index]
                # ensure enough file stats exist already
                for _ in range(len(file_cedits), len(file_lines) + 1):
                    file_cedits.append(Edits(logger=self.logger, histogram=histogram))
                    file_wedits.append(Edits(logger=self.logger))
                # ensure enough global stats exist already
                for _ in range(len(self.cedits), len(file_lines) + 1):
                    self.cedits.append(Edits(logger=self.logger, histogram=histogram))
                    self.wedits.append(Edits(logger=self.logger))
                    self.caligners.append(Alignment(logger=self.logger, confusion=bool(confusion)))
                    self.waligners.append(Alignment(logger=self.logger))
        else:
            file_lines = [{} for _ in self.input_file_grps] # line dicts for this file
            for i, pcgts in enumerate(input_pcgts):
                if pcgts is None:
                    # file/page was not found in this group
                    continue
                file_lines[i] = page_get_lines(pcgts, match, logger=self.logger)
        # compare lines with GT:
        report = dict()
        gt_lines = file_lines[0]
        for line_id in gt_lines:
            for i, input_lines in enumerate(file_lines):
                if i == 0:
                    continue
                pair = self.input_pair(i)
                report.setdefault(pair, dict()).setdefault('lines', list())
                if not input_lines:
                    # file/page was not found in this group
                    continue
                elif line_id not in input_lines:
                    self.logger.error('line "%s" in file "%s" is missing from input %d on page %s',
                                      line_id, input_files[0].ID if match == 'index'
                                      else input_files[i].ID, i, page_id)
                    report[pair]['lines'].append({line_id: 'missing'})
                    continue
                gt_line = gt_lines[line_id]
                gt_len = len(gt_line)
                gt_words = splitwords(gt_line)
                ocr_line = input_lines[line_id]
                ocr_len = len(ocr_line)
                ocr_words = splitwords(ocr_line)
                if 0.2 * (gt_len + ocr_len) < math.fabs(gt_len - ocr_len) > 5:
                    self.logger.warning('line "%s" in file "%s" from input %d deviates significantly'
                                        'in length (%d vs %d) on page %s', line_id,
                                        input_files[0].ID if match == 'index'
                                        else input_files[i].ID, i, gt_len, ocr_len, page_id)
                if metric == 'Levenshtein-fast':
                    # not exact (but fast): codepoints
                    cdist, clen = self.caligners[i].get_levenshtein_distance(ocr_line, gt_line)
                    wdist, wlen = self.waligners[i].get_levenshtein_distance(ocr_words, gt_words)
                else:
                    # exact (but slow): grapheme clusters
                    cdist, clen = self.caligners[i].get_adjusted_distance(
                        ocr_line, gt_line,
                        # Levenshtein / NFC / NFKC / historic_latin
                        normalization=metric,
                        gtlevel=gtlevel)
                    wdist, wlen = self.waligners[i].get_adjusted_distance(
                        ocr_words, gt_words,
                        # Levenshtein / NFC / NFKC / historic_latin
                        normalization=metric,
                        gtlevel=gtlevel)
                # align and accumulate edit counts for lines:
                file_cedits[i].add(cdist, clen, ocr_line, gt_line, name=line_id)
                file_wedits[i].add(wdist, wlen, ocr_words, gt_words, name=line_id)
                # todo: maybe it could be useful to retrieve and store the alignments, too
                report[pair]['lines'].append({line_id: {
                    'char-length': gt_len,
                    'char-error-rate': cdist / clen if clen else 0,
                    'word-error-rate': wdist / wlen if wlen else 0,
                    'gt': gt_line,
                    'ocr': ocr_line}})

        # report results for file
        for i, input_lines in enumerate(file_lines):
            if i == 0:
                continue
            elif not input_lines:
                # file/page was not found in this group
                continue
            pair = self.input_pair(i)
            name = self.input_name(i)
            self.logger.info("%5d lines %.3f±%.3f CER %.3f±%.3f WER %s / %s",
                             file_cedits[i].steps,
                             file_cedits[i].mean, math.sqrt(file_cedits[i].varia),
                             file_wedits[i].mean, math.sqrt(file_wedits[i].varia),
                             page_id, pair)
            self.report_pair(report, i, file_cedits, file_wedits)
            # accumulate edit counts for files
            if match == 'index':
                name_prefix = name + ":" + input_files[0].ID + ":"
            else:
                name_prefix = name + ":" + input_files[i].ID + ":"
            self.cedits[i].merge(file_cedits[i], name_prefix=name_prefix)
            self.wedits[i].merge(file_wedits[i], name_prefix=name_prefix)

        # write back result to page report
        file_path = os.path.join(self.output_file_grp, output_file_id + '.json')
        self.workspace.add_file(
            ID=output_file_id,
            file_grp=self.output_file_grp,
            pageId=input_file.pageId,
            local_filename=file_path,
            mimetype='application/json',
            content=json.dumps(report, indent=2, ensure_ascii=False))

def _linekey(line, match_on, logger=None):
    if logger is None:
        logger = getLogger('ocrd.processor.EvaluateLines')
    if match_on == 'id':
        return line.id
    if match_on == 'baseline':
        if line.Baseline is None:
            logger.error("cannot extract baseline from line '%s'", line.id)
            return line.Coords.points
        return line.Baseline.points
    if match_on == 'coords':
        return line.Coords.points

def page_get_lines(pcgts, match_on, logger=None):
    '''Get all TextLines in the page.
    
    Iterate the element hierarchy of the page `pcgts` down
    to the TextLine level. For each line, store the element
    ID and its first TextEquiv annotation.
    
    Return the stored dictionary.
    '''
    if logger is None:
        logger = getLogger('ocrd.processor.EvaluateLines')
    result = dict()
    regions = pcgts.get_Page().get_AllRegions(classes=['Text'], order='reading-order')
    if not regions:
        logger.warning("Page contains no text regions")
    for region in regions:
        lines = region.get_TextLine()
        if not lines:
            logger.warning("Region '%s' contains no text lines", region.id)
            continue
        for line in lines:
            textequivs = line.get_TextEquiv()
            if not textequivs:
                logger.warning("Line '%s' contains no text results", line.id)
                continue
            if match_on == 'index':
                for i, textequiv in enumerate(textequivs):
                    index = textequiv.index or i
                    lined = result.setdefault(line.id, dict())
                    if index in lined:
                        logger.warning("Line '%s' contains TextEquiv with and without @index", line.id)
                    lined[index] = textequiv.Unicode
            else:
                result[_linekey(line, match_on, logger=logger)] = textequivs[0].Unicode
    return result
    
