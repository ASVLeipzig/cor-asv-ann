from __future__ import absolute_import

import os
import math
import json
import itertools
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
from ocrd_models.ocrd_page import to_xml

from .config import OCRD_TOOL
from ..lib.alignment import Alignment, Edits
from .transcode import page_update_higher_textequiv_levels

TOOL_NAME = 'ocrd-cor-asv-ann-align'

@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann_align(*args, **kwargs):
    return ocrd_cli_wrap_processor(AlignLines, *args, **kwargs)

class AlignLines(Processor):
    
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL_NAME]
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)
        
    def process(self):
        """Align textlines of multiple file groups and choose the 'best' characters.
        
        Find files in all input file groups of the workspace for the same
        pageIds.
        
        Open and deserialise PAGE input files, then iterate over the element
        hierarchy down to the TextLine level, looking at each first TextEquiv.
        Align character sequences in all pairs of lines for the same TextLine IDs,
        and for each position pick the 'best' character hypothesis among the inputs.
        
        \b
        Choice depends on ``method``:
        - if `majority`, then use a majority rule over the inputs
          (requires at least 3 input fileGrps),
        - if `confidence`, then use the candidate with the highest confidence
          (requires input with per-character or per-line confidence annotations),
        - if `combined`, then try a heuristic combination of both approaches
          (requires both conditions).
        
        Then concatenate those character choices to new TextLines (without
        segmentation at lower levels).
        
        Finally, make the parent regions (higher levels) consistent with that
        textual result (via concatenation joined by whitespace).
        
        Produce new output files by serialising the resulting hierarchy.
        """
        #assert_file_grp_cardinality(self.input_file_grp, >=1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        LOG = getLogger('processor.AlignLines')

        method = self.parameter['method']
        
        ifgs = self.input_file_grp.split(",") # input file groups
        ninputs = len(ifgs)
        if ninputs < 2:
            raise Exception("need multiple input file groups to align")
        if ninputs < 3 and method in ['majority', 'combined']:
            raise Exception("need more than 2 input file groups to align in this mode")
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE) # input file tuples

        self.aligner = Alignment(logger=LOG)
        
        # get input files:
        for ift in ifts:
            # get input lines:
            file_line2seq = [{} for _ in ifgs] # line dicts for this file
            file_id2line = [{} for _ in ifgs] # line ID dicts for this file
            for i, input_file in enumerate(ift):
                if not input_file:
                    # file/page was not found in this group
                    continue
                LOG.info("INPUT FILE for %s: %s", ifgs[i], input_file.ID)
                pcgts = page_from_file(self.workspace.download_file(input_file))
                file_line2seq[i] = page_get_line_sequences(pcgts)
                file_id2line[i] = {line.id: line
                                   for line in file_line2seq[i]}
                if not i:
                    # first input fileGrp becomes base for output fileGrp
                    output_pcgts = pcgts
                    self.add_metadata(output_pcgts)

            for line_id in file_id2line[0]:
                # get line objects among all input files for this line ID
                lines = [id2line.get(line_id, None) for id2line in file_id2line]
                line0 = lines[0]
                # get line texts among all input files for this line ID
                seqs = [line2seq[line] for line, line2seq in zip(lines, file_line2seq)
                        # ignore missing lines and empty lines
                        if line in line2seq and line2seq[line][0]]
                nseqs = len(seqs)
                if not nseqs:
                    continue
                charseqs, confseqs = zip(*seqs)
                for seq in charseqs:
                    LOG.debug("next input line for '%s': %s", line_id, seq)
                # align line texts pairwise
                alignments = dict()
                distances = dict()
                for i, charseq1 in enumerate(charseqs):
                    for j, charseq2 in enumerate(charseqs[i+1:], i+1):
                        disti = distances.setdefault(i, dict())
                        aligni = alignments.setdefault(i, dict())
                        disti[j], aligni[j] = self.aligner.get_adjusted_distance(
                            charseq1, charseq2,
                            normalization=None, gtlevel=1,
                            return_alignment=True)
                        distj = distances.setdefault(j, dict())
                        alignj = alignments.setdefault(j, dict())
                        distj[i], alignj[i] = disti[j], [(y, x) for x, y in aligni[j]]
                # find min-dist path through all input files (travelling salesman)
                paths = list(itertools.permutations(range(nseqs)))
                dists = [sum(distances[i][j] for i, j in pairwise(path))
                         for path in paths]
                path = paths[min(enumerate(dists), key=lambda x: x[1])[0]]
                #LOG.debug("best path through alignments for '%s' between all input files: %s", line_id, str(path))
                # iteratively expand 2-alignments to N-alignments
                chars = list() # as sequence of tuples of alternative chars/strings
                confs = list() # as sequence of tuples of corresponding confidences
                for i, j in pairwise(path):
                    if not len(chars):
                        # init list and put into pos i and j
                        starti = startj = 0
                        for ci, cj in alignments[i][j]:
                            if ci == 0:
                                ci = ''
                            if cj == 0:
                                cj = ''
                            endi = starti + len(ci)
                            endj = startj + len(cj)
                            assert charseqs[i][starti:endi] == ci
                            assert charseqs[j][startj:endj] == cj
                            subchars = [''] * nseqs
                            subconfs = [1.0] * nseqs
                            subchars[i] = ci
                            subchars[j] = cj
                            subconfs[i] = avg(confseqs[i][starti:endi])
                            subconfs[j] = avg(confseqs[j][startj:endj])
                            starti = endi
                            startj = endj
                            chars.append(subchars)
                            confs.append(subconfs)
                    else:
                        # extend j from already existing side i
                        starti = startj = 0
                        newpos = oldpos = 0
                        while newpos < len(alignments[i][j]):
                            ci, cj = alignments[i][j][newpos]
                            if ci == 0:
                                ci = ''
                            if cj == 0:
                                cj = ''
                            endi = starti + len(ci)
                            endj = startj + len(cj)
                            assert charseqs[i][starti:endi] == ci
                            assert charseqs[j][startj:endj] == cj
                            if oldpos == len(chars):
                                # previous alignments were all shorter
                                assert not ci
                                chars[oldpos-1][j] += cj
                                confs[oldpos-1][j] = avg([confs[oldpos-1][j]] + confseqs[j][startj:endj])
                                newpos += 1
                                startj = endj
                                continue
                            subchars = chars[oldpos]
                            subconfs = confs[oldpos]
                            # start of subchars[i] == start of ci
                            if len(ci) > len(subchars[i]):
                                #LOG.debug("at %d: merging chars for '%s'←'%s'", oldpos, ci, subchars[i])
                                # merge chars and confs oldpos/oldpos+1
                                assert oldpos + 1 < len(chars)
                                nextsubchars = chars[oldpos + 1]
                                nextsubconfs = confs[oldpos + 1]
                                chars[oldpos] = ['' + c1 + c2 for c1, c2 in zip(subchars, nextsubchars)]
                                confs[oldpos] = [avg([c1, c2]) for c1, c2 in zip(subconfs, nextsubconfs)]
                                chars = chars[:oldpos+1] + chars[oldpos+2:]
                                confs = confs[:oldpos+1] + confs[oldpos+2:]
                            elif len(ci) < len(subchars[i]):
                                #LOG.debug("at %d: merging aligns for '%s'→'%s'", oldpos, ci, subchars[i])
                                # merge alignments newpos/newpos+1
                                assert newpos + 1 < len(alignments[i][j])
                                nextci, nextcj = alignments[i][j][newpos+1]
                                if nextci == 0:
                                    nextci = ''
                                if nextcj == 0:
                                    nextcj = ''
                                ci += nextci
                                cj += nextcj
                                endi += len(nextci)
                                endj += len(nextcj)
                                assert charseqs[i][starti:endi] == ci
                                assert charseqs[j][startj:endj] == cj
                                alignments[i][j][newpos] = ci, cj
                                alignments[i][j] = alignments[i][j][:newpos+1] + alignments[i][j][newpos+2:]
                            else:
                                #LOG.debug("at %d: advancing for '%s'", oldpos, ci)
                                subchars[j] = cj
                                subconfs[j] = avg(confseqs[j][startj:endj])
                                starti = endi
                                startj = endj
                                newpos += 1
                                oldpos += 1
                        assert newpos == len(alignments[i][j])
                        #assert oldpos == len(chars) # chars can be longer if previous pairs had trailing inserts
                    assert starti == len(charseqs[i])
                    assert startj == len(charseqs[j])
                # vote
                linetext = ''
                lineconf = []
                bestpath = []
                for subchars, subconfs in zip(chars, confs):
                    if method == 'majority':
                        counts = [subchars.count(subchar) for subchar in subchars]
                        best = counts.index(max(counts))
                        linetext += subchars[best]
                        lineconf.append(max(conf for count, conf in zip(counts, subconfs)
                                            if count == max(counts)))
                        bestpath.append(best)
                    elif method == 'confidence':
                        best = max(enumerate(subconfs), key=lambda x: x[1])[0]
                        linetext += subchars[best]
                        lineconf.append(subconfs[best])
                        bestpath.append(best)
                    else:
                        scores = dict()
                        for subchar, subconf in zip(subchars, subconfs):
                            scores[subchar] = subconf + scores.setdefault(subchar, 0)
                        best = max(scores, key=scores.get)
                        linetext += best
                        lineconf.append(max(conf for char, conf in zip(subchars, subconfs)
                                            if char == best))
                        bestpath.append(subchars.index(best))
                LOG.debug("best path through voting results for '%s': %s", line_id, str(bestpath))
                LOG.debug("best voted line for '%s': %s", line_id, linetext)
                lineconf = sum(lineconf) / len(lineconf)
                # write back to line0
                line0.TextEquiv[0].Unicode = linetext
                line0.TextEquiv[0].conf = lineconf

            # make higher levels consistent again:
            page_update_higher_textequiv_levels('line', output_pcgts)
            
            # write back result to page report
            file_id = make_file_id(ift[0], self.output_file_grp)
            output_pcgts.set_pcGtsId(file_id)
            file_path = os.path.join(self.output_file_grp, file_id + '.xml')
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=ift[0].pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(output_pcgts))

def page_get_line_sequences(pcgts):
    '''Get all TextLines in the page.
    
    Iterate the element hierarchy of the page `pcgts` down
    to the TextLine level. For each line element, store
    a tuple of its first TextEquiv/Unicode string and its
    per-character confidence sequence (projected from the
    line-level confidence, or preferably, if available, the
    word-level or glyph-level confidences).
    
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
            line_equivs = line.get_TextEquiv()
            if not line_equivs:
                LOG.warning("Line '%s' contains no text results", line.id)
                continue
            line_str = line_equivs[0].Unicode
            line_conf0 = float(line_equivs[0].conf or "1.0")
            line_confs = list()
            words = line.get_Word()
            for word in words:
                word_equivs = word.get_TextEquiv()
                if not word_equivs:
                    line_confs = list()
                    break
                word_conf0 = float(word_equivs[0].conf or line_conf0)
                word_confs = list()
                for glyph in word.get_Glyph():
                    glyph_equivs = glyph.get_TextEquiv()
                    if not glyph_equivs:
                        word_confs = list()
                        break
                    glyph_conf0 = float(glyph_equivs[0].conf or word_conf0)
                    glyph_str = glyph_equivs[0].Unicode
                    word_confs.extend([glyph_conf0] * len(glyph_str))
                if not len(word_confs):
                    word_str = word_equivs[0].Unicode
                    word_confs = ([word_conf0] * len(word_str))
                line_confs.extend(word_confs)
                if word is not words[-1]:
                    line_confs.append(line_conf0)
            if not len(line_confs):
                line_confs = [line_conf0] * len(line_str)
            assert len(line_confs) == len(line_str)
            result[line] = (line_str, line_confs)
    return result

# only Python 3.10
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def avg(iterable):
    l = len(iterable)
    s = sum(iterable)
    if l > 0:
        return s / l
    return 0
