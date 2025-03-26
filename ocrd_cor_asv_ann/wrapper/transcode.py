from __future__ import absolute_import

import os.path
from functools import reduce
from typing import Optional

import numpy as np
import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd import Processor, OcrdPageResult
from ocrd_utils import (
    getLogger,
    tf_disable_interactive_logs,
    xywh_from_points,
    points_from_xywh,
)
from ocrd_models.ocrd_page import (
    OcrdPage,
    GlyphType,
    WordType,
    TextLineType,
    CoordsType,
    TextEquivType,
    ReadingOrderType,
    RegionRefType,
    RegionRefIndexedType,
    OrderedGroupType,
    OrderedGroupIndexedType,
    UnorderedGroupType,
    UnorderedGroupIndexedType,
)
from ocrd_models.ocrd_page_generateds import (
    ReadingDirectionSimpleType,
    TextLineOrderSimpleType,
    TextTypeSimpleType
)


@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann_process(*args, **kwargs):
    return ocrd_cli_wrap_processor(ANNCorrection, *args, **kwargs)

class ANNCorrection(Processor):
    max_workers = 1 # cannot share TF/Keras model across fork
    
    @property
    def executable(self) -> str:
        return 'ocrd-cor-asv-ann-process'

    @property
    def metadata_filename(self) -> str:
        return os.path.join('wrapper', 'ocrd-tool.json')

    def setup(self) -> None:
        tf_disable_interactive_logs()
        model_file = self.resolve_resource(self.parameter['model_file'])
        from ..lib.seq2seq import Sequence2Sequence
        self.s2s = Sequence2Sequence(logger=self.logger, progbars=True)
        self.s2s.load_config(model_file)
        self.s2s.configure()
        self.s2s.load_weights(model_file)
        self.s2s.rejection_threshold = self.parameter['rejection_threshold']
        self.s2s.beam_width_in = self.parameter['fixed_beam_width']
        self.s2s.beam_threshold_in = self.parameter['relative_beam_width']
        self.logger.debug("Loaded model_file '%s'", self.parameter['model_file'])
        
    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Perform OCR post-correction with encoder-attention-decoder ANN on the workspace.
        
        Open and deserialise PAGE input files, then iterate over the element hierarchy
        down to the requested `textequiv_level`, making sequences of TextEquiv objects
        as lists of lines. Concatenate their string values, obeying rules of implicit
        whitespace, and map the string positions where the objects start.
        
        Next, transcode the input lines into output lines in parallel, and use
        the retrieved soft alignment scores to calculate hard alignment paths
        between input and output string via Viterbi decoding. Then use those
        to map back the start positions and overwrite each TextEquiv with its
        new content, paying special attention to whitespace:
        
        Distribute edits such that whitespace objects cannot become more than whitespace
        (or be deleted) and that non-whitespace objects must not start or end with
        whitespace (but may contain new whitespace in the middle).
        
        Subsequently, unless processing on the `line` level, make the Word segmentation
        consistent with that result again: merge around deleted whitespace tokens and
        split at whitespace inside non-whitespace tokens.
        
        Finally, make the levels above `textequiv_level` consistent with that
        textual result (via concatenation joined by whitespace).
        
        Produce new output files by serialising the resulting hierarchy.
        """
        pcgts = input_pcgts[0]
        level = self.parameter['textequiv_level']
        self.logger.info("Correcting text in page '%s' at the %s level", page_id, level)

        # get textequiv references for all lines:
        line_sequences = _page_get_line_sequences_at(level, pcgts, logger=self.logger)

        # concatenate to strings and get dict of start positions to refs:
        input_lines, textequiv_starts = \
            _line_sequences2confmat_sequences(
                self.s2s.mapping[0], line_sequences,
                charmap=self.parameter['charmap'])

        # correct string and get input-output alignment:
        # FIXME: split into self.batch_size chunks
        output_lines, output_probs, output_scores, alignments = \
            self.s2s.correct_lines(
                input_lines, conf=input_lines,
                fast=self.parameter['fast_mode'],
                greedy=self.parameter['fast_mode'])

        # re-align (from alignment scores) and overwrite the textequiv references:
        for (input_line, output_line, output_prob,
             output_score, alignment, textequivs) in zip(
                 input_lines, output_lines, output_probs,
                 output_scores, alignments, textequiv_starts):
            input_line_top = ''.join([chunk[0][0] for chunk in input_line])
            input_line_len = sum((max((len(x[0]) for x in chunk), default=0) for chunk in input_line))

            # convert soft scores (seen from output) to hard path (seen from input):
            realignment, distance = _alignment2path(alignment, input_line_len, len(output_line),
                                                    1. / self.s2s.voc_size)
            # create hard path via minimal edit distance:
            # (cannot be applied directly if input_line is in confmat format)
            #realignment, distance = _alignment_path(input_line, output_line)

            # overwrite TextEquiv references:
            line, new_sequence = _update_sequence(
                input_line, output_line, output_prob, output_score,
                realignment, textequivs, self.logger)

            # update Word segmentation:
            if level != 'line':
                _resegment_sequence(new_sequence, level, logger=self.logger)

            if input_line_top != output_line:
                self.logger.debug('"%s" → "%s"', input_line_top.rstrip('\n'), output_line.rstrip('\n'))
                self.logger.info('corrected line "%s" with %d elements, ppl: %.3f, CER: %.1f%%',
                                 line.id,
                                 len([x for x in new_sequence if x.index != -1]),
                                 np.exp(output_score),
                                 distance / len(realignment) * 100)
            else:
                self.logger.debug('"%s"', input_line_top.rstrip('\n'))
                self.logger.info('kept line "%s"', line.id)

        # make higher levels consistent again:
        if level != 'region':
            page_update_higher_textequiv_levels(level, pcgts)
        # remove lower levels (cannot be made consistent):
        if level != 'glyph':
            page_remove_lower_textequiv_levels(level, pcgts)

        return OcrdPageResult(pcgts)
            
def _page_get_line_sequences_at(level: str, pcgts: OcrdPage, logger=None):
    '''Get TextEquiv sequences for PAGE-XML hierarchy level including whitespace.
    
    Return a list of lines from the document `pcgts`, where
    each line is a list of TextEquiv lists from the given
    hierarchy `level`. This includes artificial TextEquiv for
    implicit whitespace between elements (marked by `index=-1`,
    which is forbidden in the XML Schema).

    Thus, for `level=line`, each line is a list of the line's
    TextEquiv list followed by a newline. For `level=word` or `glyph`,
    each line is a list of words, interspersed by spaces and
    followed by a newline.
    '''
    if logger is None:
        logger = getLogger('ocrd.processor.ANNCorrection')
    sequences = []
    regions = pcgts.get_Page().get_AllRegions(classes=['Text'], order='reading-order')
    if not regions:
        logger.warning("Page contains no text regions")
    for region in regions:
        lines = region.get_TextLine()
        if not lines:
            logger.warning("Region '%s' contains no text lines", region.id)
        for line in lines:
            sequences.append([])
            if level == 'line':
                logger.log(5, "Getting text in line '%s'", line.id)
                textequivs = line.get_TextEquiv()
                if len(textequivs):
                    sequences[-1].append(textequivs)
                else:
                    logger.warning("Line '%s' contains no text results", line.id)
            else:
                words = line.get_Word()
                if not words:
                    logger.warning("Line '%s' contains no word", line.id)
                    continue # no EOL
                for word in words:
                    if level == 'word':
                        logger.log(5, "Getting text in word '%s'", word.id)
                        textequivs = word.get_TextEquiv()
                        if textequivs:
                            sequences[-1].append(textequivs)
                        else:
                            logger.warning("Word '%s' contains no text results", word.id)
                            continue # no inter-word
                    else:
                        glyphs = word.get_Glyph()
                        if not glyphs:
                            logger.warning("Word '%s' contains no glyphs", word.id)
                            continue # no inter-word
                        for glyph in glyphs:
                            logger.log(5, "Getting text in glyph '%s'", glyph.id)
                            textequivs = glyph.get_TextEquiv()
                            if len(textequivs):
                                sequences[-1].append(textequivs)
                            else:
                                logger.warning("Glyph '%s' contains no text results", glyph.id)
                                # treat as gap
                                gap = TextEquivType(Unicode='', conf=1.0)
                                gap.parent_object_ = glyph
                                glyph.set_TextEquiv([gap])
                                sequences[-1].append([gap])
                    space = TextEquivType(Unicode=' ', conf=1.0, index=-1)
                    space.parent_object_ = word if level == 'word' else glyph
                    sequences[-1].append([space])
                if len(sequences[-1]):
                    sequences[-1].pop() # no inter-word
            newline = TextEquivType(Unicode='\n', conf=1.0, index=-1)
            newline.parent_object_ = line if level == 'line' else word if level == 'word' else glyph
            sequences[-1].append([newline])
    # filter empty lines (containing only newline):
    return [line for line in sequences if len(line) > 1]

def _line_sequences2confmat_sequences(mapping, line_sequences, charmap=None):
    '''Concatenate TextEquiv sequences to line strings.
    
    \b
    Return a 2-tuple:
    - a list of lines as confmats (where a confmat is a horizontal list of
      vertical lists of alternative+confidence tuples)
    - a list of lines as dicts from string positions to TextEquiv references.
    
    If `charmap` is not None, apply it as a character translation
    table on all characters.
    '''
    from ..lib.seq2seq import GAP
    if charmap:
        charmap = str.maketrans(charmap)
    input_lines, textequiv_starts = [], []
    for line_sequence in line_sequences:
        i = 0
        input_lines.append([])
        textequiv_starts.append({})
        for textequivs in line_sequence:
            textequiv_starts[-1][i] = textequivs
            for textequiv in textequivs:
                if charmap:
                    textequiv.Unicode = textequiv.Unicode.translate(charmap)
                if not textequiv.Unicode:
                    # empty element (OCR rejection):
                    # this information is still valuable for post-correction,
                    # and we reserved index zero for underspecified inputs,
                    # therefore here we just need to replace the gap with some
                    # unmapped character, like GAP:
                    assert GAP not in mapping, (
                        'character "%s" must not be mapped (needed for gap repair)' % GAP)
                    textequiv.Unicode = GAP
                if textequiv.conf is None:
                    textequiv.conf = 1.0
            # confmat input uses zero-padding within sequences
            # so we need the longest alternative:
            j = max((len(textequiv.Unicode) for textequiv in textequivs), default=0)
            input_lines[-1].append([(textequiv.Unicode, textequiv.conf) for textequiv in textequivs])
            i += j
    return input_lines, textequiv_starts

def _alignment2path(alignment, i_max, j_max, min_score):
    '''Find the best path through a soft alignment matrix via Viterbi search.
    
    The `alignment` is a list of vectors of scores (between 0..1).
    The list indexes are output positions (ignored above `j_max`),
    the vector indexes are input positions (ignored above `i_max`).
    Viterbi forward scores are only calculated where the alignment
    scores are larger than `min_score` (to save time).
    
    Return a dictionary mapping input positions to output positions
    (i.e. a realignment path).
    '''
    # compute Viterbi forward pass:
    viterbi_fw = np.zeros((i_max, j_max), dtype=np.float32)
    dist = 0
    i, j = 0, 0
    while i < i_max and j < j_max:
        if i > 0:
            im1 = viterbi_fw[i - 1, j]
        else:
            im1 = 0
        if j > 0:
            jm1 = viterbi_fw[i, j - 1]
        else:
            jm1 = 0
        if i > 0 and j > 0:
            ijm1 = viterbi_fw[i - 1, j - 1]
        else:
            ijm1 = 0
        viterbi_fw[i, j] = alignment[j][i] + max(im1, jm1, ijm1)
        while True:
            i += 1
            if i == i_max:
                j += 1
                if j == j_max:
                    break
                i = 0
            if alignment[j][i] > min_score:
                break
    # compute Viterbi backward pass:
    i = i_max - 1 if i_max <= j_max else j_max - 2 + int(
        np.argmax(viterbi_fw[j_max - i_max - 2:, j_max - 1]))
    j = j_max - 1 if j_max <= i_max else i_max - 2 + int(
        np.argmax(viterbi_fw[i_max - 1, i_max - j_max - 2:]))
    realignment = {i_max: j_max} # init end of line
    while i >= 0 and j >= 0:
        dist += 1.0 - alignment[j][i]
        realignment[i] = j # (overwrites any previous assignment)
        if viterbi_fw[i - 1, j] > viterbi_fw[i, j - 1]:
            if viterbi_fw[i - 1, j] > viterbi_fw[i - 1, j - 1]:
                i -= 1
            else:
                i -= 1
                j -= 1
        elif viterbi_fw[i, j - 1] > viterbi_fw[i - 1, j - 1]:
            j -= 1
        else:
            j -= 1
            i -= 1
    realignment[0] = 0 # init start of line
    # logger = getLogger('ocrd.processor.ANNCorrection')
    # logger.debug('realignment: %s', str(realignment))
    # from matplotlib import pyplot
    # pyplot.subplot(2, 1, 1)
    # pyplot.imshow(np.array(alignment).T)
    # pyplot.title("alignment")
    # pyplot.subplot(2, 1, 2)
    # pyplot.imshow(viterbi_fw)
    # pyplot.title("Viterbi forward")
    # pyplot.show()
    return realignment, dist

def _alignment_path(input_text, output_text):
    '''Find the minimal distance path through string pair via Smith-Waterman alignment.
    
    Return a dictionary mapping input positions to output positions
    (i.e. a realignment path).
    '''
    from ..lib.alignment import Alignment
    alignment = Alignment.best_alignment(input_text, output_text)
    realignment = {0: 0} # init start of line
    i, j, dist = 0, 0, 0.0
    for input_sym, output_sym in alignment:
        if input_sym:
            i += len(input_sym)
        if output_sym:
            j += len(output_sym)
        if input_sym != output_sym:
            dist += 1.0
        realignment[i] = j
    assert i == len(input_text)
    assert j == len(output_text)
    assert len(alignment) > 0
    dist /= len(alignment) - 1 # ignore newline
    # logger = getLogger('ocrd.processor.ANNCorrection')
    # logger.debug('realignment: %s', str(realignment))
    return realignment, dist

def _update_sequence(input_sequence, output_line, output_prob,
                     score, realignment, textequiv_starts, logger=None):
    '''Apply correction across TextEquiv elements along alignment path of one line.
    
    Traverse the path `realignment` through `input_line` and `output_line`,
    looking up TextEquiv objects by their start positions via `textequivs`.
    Overwrite the string value of the objects (which equals the segment in
    `input_line`) with the corrected version (which equals the segment in
    `output_line`), and overwrite the confidence values from `output_prob`.
    
    Also, redistribute string parts bordering whitespace: make sure space
    only maps to space (or gets deleted, which necessitates merging Words),
    and non-space only maps to non-space (with space allowed only in the
    middle, which necessitates splitting Words). This is required in order
    to avoid loosing content: the implicit whitespace TextEquivs do not
    belong to the document hierarchy itself.
    (Merging and splitting can be done afterwards.)

    \b
    Return a tuple:
    - the TextLine object (parent of all TextEquivs)
    - the sequence of resulting TextEquivs (without alternatives)
    '''
    if logger is None:
        logger = getLogger('ocrd.processor.ANNCorrection')
    input_line = '' # concatenation of longest alternatives of each chunk
    for chunk in input_sequence:
        input_line += sorted([x[0] for x in chunk], key=len)[-1]
    i_max = len(input_line)
    j_max = len(output_line)
    textequiv_starts.setdefault(i_max, None) # init end of line
    line = textequiv_starts[0][0].parent_object_
    if isinstance(line, GlyphType):
        line = line.parent_object_
    if isinstance(line, WordType):
        line = line.parent_object_
    assert isinstance(line, TextLineType), line
    last = []
    sequence = []
    for i in textequiv_starts:
        if i in realignment:
            j = realignment[i]
        else:
            # this element was deleted
            j = last[1]
        logger.log(5, f"last={last}, [i,j]={[i, j]}")
        if last:
            input_ = input_line[last[0]:i]
            output = output_line[last[1]:j]
            prob = output_prob[last[1]:j]
            textequivs = textequiv_starts[last[0]]
            unicodes = [textequiv.Unicode for textequiv in textequivs]
            assert input_ in unicodes, (
                'no source element alternative "%s" matches input section "%s" in line "%s"' % (
                    str(unicodes), input_, line.id))
            # select the first textequiv (others will be removed afterwards)
            textequiv = textequivs[0]
            logger.log(5, f"{repr(input_)} → {repr(output)}"
                       f" [{str(textequiv.conf if textequiv.index != -1 else -1)} → {prob}]")
            # try to distribute whitespace onto whitespace, i.e.
            # if input is Whitespace, move any Non-whitespace parts
            # in output to neighbours;
            # otherwise, move Whitespace parts to neighbours
            # if their input is Whitespace too;
            # input:  N|    W    |N   N|     W   |   W|    N    |W
            # output:  |<-N W N->|     |<-W<-N W |    |<-W N W->|
            if textequiv.index == -1:
                if output and not output.startswith((" ", "\n")) and len(sequence):
                    while output and not output.startswith((" ", "\n")):
                        sequence[-1].Unicode += output[0]
                        last[1] += 1
                        output = output[1:]
                    logger.log(5, 'corrected non-whitespace LHS')
                if output and not output.endswith((" ", "\n")):
                    j -= len(output.split(" ")[-1])
                    output = output_line[last[1]:j]
                    logger.log(5, 'corrected non-whitespace RHS')
                if output.split() and len(sequence):
                    while output.split():
                        sequence[-1].Unicode += output[0]
                        last[1] += 1
                        output = output[1:]
                    logger.log(5, 'corrected non-whitespace middle')
            else:
                if output.startswith(" ") and len(sequence) and sequence[-1].index == -1:
                    while output.startswith(" "):
                        sequence[-1].Unicode += output[0]
                        last[1] += 1
                        output = output[1:]
                    logger.log(5, 'corrected whitespace LHS')
                if output.endswith((" ", "\n")) and i < i_max and textequiv_starts[i][0].index == -1:
                    while output.endswith((" ", "\n")):
                        j -= 1
                        output = output[:-1]
                    logger.log(5, 'corrected whitespace RHS')
            textequiv.Unicode = output
            #textequiv.conf = np.exp(-score)
            textequiv.conf = np.mean(prob or [1.0])
            sequence.append(textequiv)
        last = [i, j]
    assert last == [i_max, j_max], (
        'alignment path did not reach top: %d/%d vs %d/%d in line "%s"' % (
            last[0], last[1], i_max, j_max, line.id))
    for i, textequiv in enumerate(sequence):
        # disallow non-whitespace mapping to -1 (artificial whitespace)
        # (but allow whitespace to anything and anything to true segments)
        assert not textequiv.Unicode.split() or textequiv.index != -1, (
            'output "%s" will be lost at (whitespace) element %d in line "%s"' % (
                textequiv.Unicode, i, line.id))
    return line, sequence

def _resegment_sequence(sequence, level, logger=None):
    '''Merge and split Words among `sequence` after correction.
    
    At each empty whitespace TextEquiv, merge the neighbouring Words.
    At each non-whitespace TextEquiv which contains whitespace, split
    the containing Word at the respective positions.
    '''
    if logger is None:
        logger = getLogger('ocrd.processor.ANNCorrection')
    for i, textequiv in enumerate(sequence):
        if level == 'glyph':
            word = textequiv.parent_object_.parent_object_
        else:
            word = textequiv.parent_object_
        textline = word.parent_object_
        assert isinstance(word, WordType), word
        assert isinstance(textline, TextLineType), textline
        if textequiv.index == -1:
            if not textequiv.Unicode:
                # whitespace was deleted: merge adjacent words
                if i == 0 or i == len(sequence) - 1:
                    logger.error('cannot merge Words at the %s of line "%s"',
                                 'end' if i else 'start', textline.id)
                else:
                    prev_textequiv = sequence[i - 1]
                    next_textequiv = sequence[i + 1]
                    if level == 'glyph':
                        prev_word = prev_textequiv.parent_object_.parent_object_
                        next_word = next_textequiv.parent_object_.parent_object_
                    else:
                        prev_word = prev_textequiv.parent_object_
                        next_word = next_textequiv.parent_object_
                    merged = _merge_words(prev_word, next_word)
                    logger.debug('merged %s and %s to %s in line %s',
                                 prev_word.id, next_word.id, merged.id, textline.id)
                    textline.set_Word([merged if word is prev_word else word
                                       for word in textline.get_Word()
                                       if not word is next_word])
        elif " " in textequiv.Unicode:
            # whitespace was introduced: split word
            if level == 'glyph':
                glyph = next(glyph for glyph in word.get_Glyph()
                             if textequiv in glyph.get_TextEquiv())
                prev_, next_ = _split_word_at_glyph(word, glyph)
                parts = [prev_, next_]
            else:
                parts = []
                next_ = word
                while True:
                    prev_, next_ = _split_word_at_space(next_)
                    if " " in next_.get_TextEquiv()[0].Unicode:
                        parts.append(prev_)
                    else:
                        parts.append(prev_)
                        parts.append(next_)
                        break
            logger.debug('split %s to %s in line %s',
                         word.id, [w.id for w in parts], textline.id)
            textline.set_Word(reduce(lambda l, w, key=word, value=parts:
                                     l + value if w is key else l + [w],
                                     textline.get_Word(), []))
    
def _merge_words(prev_, next_):
    merged = WordType(id=prev_.id + '.' + next_.id)
    merged.set_Coords(CoordsType(points=points_from_xywh(xywh_from_points(
        prev_.get_Coords().points + ' ' + next_.get_Coords().points))))
    if prev_.get_language():
        merged.set_language(prev_.get_language())
    if prev_.get_TextStyle():
        merged.set_TextStyle(prev_.get_TextStyle())
    if prev_.get_Glyph() or next_.get_Glyph():
        merged.set_Glyph(prev_.get_Glyph() + next_.get_Glyph())
    if prev_.get_TextEquiv():
        merged.set_TextEquiv(prev_.get_TextEquiv())
    else:
        merged.set_TextEquiv([TextEquivType(Unicode='', conf=1.0)])
    if next_.get_TextEquiv():
        textequiv = merged.get_TextEquiv()[0]
        textequiv2 = next_.get_TextEquiv()[0]
        textequiv.Unicode += textequiv2.Unicode
        if textequiv.conf and textequiv2.conf:
            textequiv.conf *= textequiv2.conf
    return merged

def _split_word_at_glyph(word, glyph):
    prev_ = WordType(id=word.id + '_l')
    next_ = WordType(id=word.id + '_r')
    xywh_glyph = xywh_from_points(glyph.get_Coords().points)
    xywh_word = xywh_from_points(word.get_Coords().points)
    xywh_prev = xywh_word.copy()
    xywh_prev.update({'w': xywh_glyph['x'] - xywh_word['x']})
    prev_.set_Coords(CoordsType(points=points_from_xywh(
        xywh_prev)))
    xywh_next = xywh_word.copy()
    xywh_next.update({'x': xywh_glyph['x'] - xywh_glyph['w'],
                      'w': xywh_word['w'] - xywh_prev['w']})
    next_.set_Coords(CoordsType(points=points_from_xywh(
        xywh_next)))
    if word.get_language():
        prev_.set_language(word.get_language())
        next_.set_language(word.get_language())
    if word.get_TextStyle():
        prev_.set_TextStyle(word.get_TextStyle())
        next_.set_TextStyle(word.get_TextStyle())
    glyphs = word.get_Glyph()
    pos = glyphs.index(glyph)
    prev_.set_Glyph(glyphs[0:pos])
    next_.set_Glyph(glyphs[pos+1:])
    # TextEquiv: will be overwritten by page_update_higher_textequiv_levels
    return prev_, next_

def _split_word_at_space(word):
    prev_ = WordType(id=word.id + '_l')
    next_ = WordType(id=word.id + '_r')
    xywh = xywh_from_points(word.get_Coords().points)
    textequiv = word.get_TextEquiv()[0]
    pos = textequiv.Unicode.index(" ")
    fract = pos / len(textequiv.Unicode)
    xywh_prev = xywh.copy()
    xywh_prev.update({'w': xywh['w'] * fract})
    prev_.set_Coords(CoordsType(points=points_from_xywh(
        xywh_prev)))
    xywh_next = xywh.copy()
    xywh_next.update({'x': xywh['x'] + xywh['w'] * fract,
                      'w': xywh['w'] * (1 - fract)})
    next_.set_Coords(CoordsType(points=points_from_xywh(
        xywh_next)))
    if word.get_language():
        prev_.set_language(word.get_language())
        next_.set_language(word.get_language())
    if word.get_TextStyle():
        prev_.set_TextStyle(word.get_TextStyle())
        next_.set_TextStyle(word.get_TextStyle())
    # Glyphs: irrelevant at this processing level
    textequiv_prev = TextEquivType(Unicode=textequiv.Unicode[0:pos],
                                   conf=textequiv.conf)
    textequiv_next = TextEquivType(Unicode=textequiv.Unicode[pos+1:],
                                   conf=textequiv.conf)
    prev_.set_TextEquiv([textequiv_prev])
    next_.set_TextEquiv([textequiv_next])
    return prev_, next_

def page_update_higher_textequiv_levels(level, pcgts, overwrite=True):
    """Update the TextEquivs of all PAGE-XML hierarchy levels above ``level`` for consistency.
    
    Starting with the lowest hierarchy level chosen for processing,
    join all first TextEquiv.Unicode (by the rules governing the respective level)
    into TextEquiv.Unicode of the next higher level, replacing them.
    If ``overwrite`` is false and the higher level already has text, keep it.
    
    When two successive elements appear in a ``Relation`` of type ``join``,
    then join them directly (without their respective white space).
    
    Likewise, average all first TextEquiv.conf into TextEquiv.conf of the next higher level.
    
    In the process, traverse the words and lines in their respective ``readingDirection``,
    the (text) regions which contain lines in their respective ``textLineOrder``, and
    the (text) regions which contain text regions in their ``ReadingOrder``
    (if they appear there as an ``OrderedGroup``).
    Where no direction/order can be found, use XML ordering.
    
    Follow regions recursively, but make sure to traverse them in a depth-first strategy.
    """
    page = pcgts.get_Page()
    relations = page.get_Relations() # get RelationsType
    if relations:
        relations = relations.get_Relation() # get list of RelationType
    else:
        relations = []
    joins = list() # 
    for relation in relations:
        if relation.get_type() == 'join': # ignore 'link' type here
            joins.append((relation.get_SourceRegionRef().get_regionRef(),
                          relation.get_TargetRegionRef().get_regionRef()))
    reading_order = dict()
    ro = page.get_ReadingOrder()
    if ro:
        page_get_reading_order(reading_order, ro.get_OrderedGroup() or ro.get_UnorderedGroup())
    if level != 'region':
        for region in page.get_AllRegions(classes=['Text']):
            # order is important here, because regions can be recursive,
            # and we want to concatenate by depth first;
            # typical recursion structures would be:
            #  - TextRegion/@type=paragraph inside TextRegion
            #  - TextRegion/@type=drop-capital followed by TextRegion/@type=paragraph inside TextRegion
            #  - any region (including TableRegion or TextRegion) inside a TextRegion/@type=footnote
            #  - TextRegion inside TableRegion
            subregions = region.get_TextRegion()
            if subregions: # already visited in earlier iterations
                # do we have a reading order for these?
                # TODO: what if at least some of the subregions are in reading_order?
                if (all(subregion.id in reading_order for subregion in subregions) and
                    isinstance(reading_order[subregions[0].id], # all have .index?
                               (OrderedGroupType, OrderedGroupIndexedType))):
                    subregions = sorted(subregions, key=lambda subregion:
                                        reading_order[subregion.id].index)
                region_unicode = page_element_unicode0(subregions[0])
                for subregion, next_subregion in zip(subregions, subregions[1:]):
                    if (subregion.id, next_subregion.id) not in joins:
                        region_unicode += '\n' # or '\f'?
                    region_unicode += page_element_unicode0(next_subregion)
                region_conf = sum(page_element_conf0(subregion) for subregion in subregions)
                region_conf /= len(subregions)
            else: # TODO: what if a TextRegion has both TextLine and TextRegion children?
                lines = region.get_TextLine()
                if ((region.get_textLineOrder() or
                     page.get_textLineOrder()) ==
                    TextLineOrderSimpleType.BOTTOMTOTOP):
                    lines = list(reversed(lines))
                if level != 'line':
                    for line in lines:
                        words = line.get_Word()
                        if ((line.get_readingDirection() or
                             region.get_readingDirection() or
                             page.get_readingDirection()) ==
                            ReadingDirectionSimpleType.RIGHTTOLEFT):
                            words = list(reversed(words))
                        if level != 'word':
                            for word in words:
                                glyphs = word.get_Glyph()
                                if ((word.get_readingDirection() or
                                     line.get_readingDirection() or
                                     region.get_readingDirection() or
                                     page.get_readingDirection()) ==
                                    ReadingDirectionSimpleType.RIGHTTOLEFT):
                                    glyphs = list(reversed(glyphs))
                                word_unicode = ''.join(page_element_unicode0(glyph) for glyph in glyphs)
                                word_conf = sum(page_element_conf0(glyph) for glyph in glyphs)
                                if glyphs:
                                    word_conf /= len(glyphs)
                                if not word.get_TextEquiv() or overwrite:
                                    word.set_TextEquiv( # replace old, if any
                                        [TextEquivType(Unicode=word_unicode, conf=word_conf)])
                        line_unicode = ' '.join(page_element_unicode0(word) for word in words)
                        line_conf = sum(page_element_conf0(word) for word in words)
                        if words:
                            line_conf /= len(words)
                        if not line.get_TextEquiv() or overwrite:
                            line.set_TextEquiv( # replace old, if any
                                [TextEquivType(Unicode=line_unicode, conf=line_conf)])
                region_unicode = ''
                region_conf = 0
                if lines:
                    region_unicode = page_element_unicode0(lines[0])
                    for line, next_line in zip(lines, lines[1:]):
                        words = line.get_Word()
                        next_words = next_line.get_Word()
                        if not(words and next_words and (words[-1].id, next_words[0].id) in joins):
                            region_unicode += '\n'
                        region_unicode += page_element_unicode0(next_line)
                    region_conf = sum(page_element_conf0(line) for line in lines)
                    region_conf /= len(lines)
            if not region.get_TextEquiv() or overwrite:
                region.set_TextEquiv( # replace old, if any
                    [TextEquivType(Unicode=region_unicode, conf=region_conf)])

def page_get_reading_order(ro, rogroup):
    """Add all elements from the given reading order group to the given dictionary.
    
    Given a dict ``ro`` from layout element IDs to ReadingOrder element objects,
    and an object ``rogroup`` with additional ReadingOrder element objects,
    add all references to the dict, traversing the group recursively.
    """
    regionrefs = list()
    if isinstance(rogroup, (OrderedGroupType, OrderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRefIndexed() +
                      rogroup.get_OrderedGroupIndexed() +
                      rogroup.get_UnorderedGroupIndexed())
    if isinstance(rogroup, (UnorderedGroupType, UnorderedGroupIndexedType)):
        regionrefs = (rogroup.get_RegionRef() +
                      rogroup.get_OrderedGroup() +
                      rogroup.get_UnorderedGroup())
    for elem in regionrefs:
        ro[elem.get_regionRef()] = elem
        if not isinstance(elem, (RegionRefType, RegionRefIndexedType)):
            page_get_reading_order(ro, elem)

def page_element_unicode0(element):
    """Get Unicode string of the first text result."""
    if element.TextEquiv:
        return element.TextEquiv[0].Unicode
    else:
        return ''

def page_element_conf0(element):
    """Get confidence (as float value) of the first text result."""
    if element.TextEquiv:
        return 1.0 if element.TextEquiv[0].conf is None else element.TextEquiv[0].conf
    return 1.0

def page_remove_lower_textequiv_levels(level, pcgts):
    page = pcgts.Page
    if level == 'region':
        for region in page.get_AllRegions(classes=['Text']):
            region.TextEquiv = []
    else:
        for line in page.get_AllTextLines():
            if level == 'line':
                line.Word = []
            else:
                for word in line.Word or []:
                    if level == 'word':
                        word.Glyph = []
                    else:
                        for glyph in word.Glyph:
                            glyph.Graphemes = []
