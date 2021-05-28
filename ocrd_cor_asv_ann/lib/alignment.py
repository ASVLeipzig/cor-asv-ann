# -*- coding: utf-8
import logging
import bisect
import unicodedata

class Alignment():
    def __init__(self, gap_element=0, logger=None, confusion=False):
        self.confusion = dict() if confusion else None
        self.gap_element = gap_element
        self.logger = logger or logging.getLogger(__name__)
        # alignment for windowing...
        ## python-alignment is impractical with long or heavily deviating sequences (see github issues 9, 10, 11):
        #import alignment.sequence
        #alignment.sequence.GAP_ELEMENT = self.gap_element # override default
        #from alignment.sequence import Sequence, gap_element
        #from alignment.vocabulary import Vocabulary
        #from alignment.sequencealigner import SimpleScoring, StrictGlobalSequenceAligner
        # Levenshtein scoring:
        #self.scoring = SimpleScoring(2,-1) # match score, mismatch score
        #self.aligner = StrictGlobalSequenceAligner(scoring,-2) # gap score
        # Levenshtein-like scoring with 0.1 distance within typical OCR confusion classes (to improve alignment quality; to reduce error introduced by windowing):
        # class OCRScoring(SimpleScoring):
        #     def __init__(self):
        #         super(OCRScoring, self).__init__(0,-1) # match score, mismatch score (Levenshtein-like)
        #         self.classes = [[u"a", u"ä", u"á", u"â", u"à", u"ã"],
        #                         [u"o", u"ö", u"ó", u"ô", u"ò", u"õ"],
        #                         [u"u", u"ü", u"ú", u"û", u"ù", u"ũ"],
        #                         [u"A", u"Ä", u"Á", u"Â", u"À", u"Ã"],
        #                         [u"O", u"Ö", u"Ó", u"Ô", u"Ò", u"Õ"],
        #                         [u"U", u"Ü", u"Ú", u"Û", u"Ù", u"Ũ"],
        #                         [0, u"ͤ"],
        #                         [u'"', u"“", u"‘", u"'", u"’", u"”"],
        #                         [u',', u'‚', u'„'],
        #                         [u'-', u'‐', u'—', u'–', u'_'],
        #                         [u'=', u'⸗', u'⹀'],
        #                         [u'ſ', u'f', u'ß'], #s?
        #                         [u"c", u"<", u"e"]]
        #         self.table = {}
        #         for c in self.classes:
        #             for i in c:
        #                 for j in c:
        #                     if i==j:
        #                         self.table[(i,j)] = 0.0
        #                     else:
        #                         self.table[(i,j)] = 0.1
        #     def __call__(self, firstElement, secondElement):
        #         if (firstElement,secondElement) in self.table:
        #             return self.table[(firstElement,secondElement)]
        #         else:
        #             return super(OCRScoring, self).__call__(firstElement, secondElement)
        #
        # self.scoring = OCRScoring()
        # self.aligner = StrictGlobalSequenceAligner(scoring,-1) # gap score
        
        ## edlib does not work on Unicode (non-ASCII strings)
        # import edlib

        ## difflib is optimised for visual comparisons (Ratcliff-Obershelp), not minimal distance (Levenshtein):
        from difflib import SequenceMatcher
        self.matcher = SequenceMatcher(isjunk=None, autojunk=False)
        
        ## edit_distance is impractical with long sequences, even if very similar (GT lines > 1000 characters, see github issue 6)
        # from edit_distance.code import SequenceMatcher # similar API to difflib.SequenceMatcher
        # def char_similar(a, b):
        #     return (a == b or (a,b) in table)
        # self.matcher = SequenceMatcher(test=char_similar)
        
        self.source_text = []
        self.target_text = []
    
    def set_seqs(self, source_text, target_text):
        ## code for python_alignment:
        #vocabulary = Vocabulary() # inefficient, but helps keep search space smaller if independent for each line
        #self.source_seq = vocabulary.encodeSequence(Sequence(source_text))
        #self.target_seq = vocabulary.encodeSequence(Sequence(target_text))
        
        ## code for edlib:
        #self.edres = edlib.align(source_text, target_text, mode='NW', task='path',
        #                         k=max(len(source_text),len(target_text))*2)
        
        ## code for difflib/edit_distance:
        self.matcher.set_seqs(source_text, target_text)
        
        self.source_text = source_text
        self.target_text = target_text
        
    
    def is_bad(self):
        ## code for python_alignment:
        #score = self.aligner.align(self.source_seq, self.target_seq)
        #if score < -10 and score < 5-len(source_text):
        #    return True
        
        ## code for edlib:
        # assert self.edres
        # if self.edres['editDistance'] < 0:
        #    return True

        ## code for difflib/edit_distance:
        # self.matcher = difflib_matcher if len(source_text) > 4000 or len(target_text) > 4000 else editdistance_matcher
        
        # if self.matcher.distance() > 10 and self.matcher.distance() > len(self.source_text)-5:
        return bool(self.matcher.quick_ratio() < 0.5 and len(self.source_text) > 5)
    
    def get_best_alignment(self):
        ## code for identity alignment (for GT-only training; faster, no memory overhead)
        # alignment1 = zip(source_text, target_text)
        
        ## code for python_alignment:
        #score, alignments = self.aligner.align(self.source_seq, self.target_seq, backtrace=True)
        #alignment1 = vocabulary.decodeSequenceAlignment(alignments[0])
        #alignment1 = zip(alignment1.first, alignment1.second)
        #print ('alignment score:', alignment1.score)
        #print ('alignment rate:', alignment1.percentIdentity())
        
        ## code for edlib:
        # assert self.edres
        # alignment1 = []
        # n = ""
        # source_k = 0
        # target_k = 0
        # for c in self.edres['cigar']:
        #     if c.isdigit():
        #         n = n + c
        #     else:
        #         i = int(n)
        #         n = ""
        #         if c in "=X": # identity/substitution
        #             alignment1.extend(zip(self.source_text[source_k:source_k+i], self.target_text[target_k:target_k+i]))
        #             source_k += i
        #             target_k += i
        #         elif c == "I": # insert into target
        #             alignment1.extend(zip(self.source_text[source_k:source_k+i], [self.gap_element]*i))
        #             source_k += i
        #         elif c == "D": # delete from target
        #             alignment1.extend(zip([self.gap_element]*i, self.target_text[target_k:target_k+i]))
        #             target_k += i
        #         else:
        #             raise Exception("edlib returned invalid CIGAR opcode", c)
        # assert source_k == len(self.source_text)
        # assert target_k == len(self.target_text)
        
        ## code for difflib/edit_distance:
        alignment1 = []
        source_end = len(self.source_text)
        target_end = len(self.target_text)
        try:
            opcodes = self.matcher.get_opcodes()
        except Exception as err:
            self.logger.exception('alignment of "%s" and "%s" failed',
                                  self.source_text, self.target_text)
            raise err
        for opcode, source_begin, source_end, target_begin, target_end in opcodes:
            if opcode == 'equal':
                alignment1.extend(zip(self.source_text[source_begin:source_end],
                                      self.target_text[target_begin:target_end]))
            elif opcode == 'replace': # not really substitution:
                delta = source_end-source_begin-target_end+target_begin
                #alignment1.extend(zip(self.source_text[source_begin:source_end] + [self.gap_element]*(-delta),
                #                      self.target_text[target_begin:target_end] + [self.gap_element]*(delta)))
                if delta > 0: # replace+delete
                    alignment1.extend(zip(self.source_text[source_begin:source_end-delta],
                                          self.target_text[target_begin:target_end]))
                    alignment1.extend(zip(self.source_text[source_end-delta:source_end],
                                          [self.gap_element]*(delta)))
                if delta <= 0: # replace+insert
                    alignment1.extend(zip(self.source_text[source_begin:source_end],
                                          self.target_text[target_begin:target_end+delta]))
                    alignment1.extend(zip([self.gap_element]*(-delta),
                                          self.target_text[target_end+delta:target_end]))
            elif opcode == 'insert':
                alignment1.extend(zip([self.gap_element]*(target_end-target_begin),
                                      self.target_text[target_begin:target_end]))
            elif opcode == 'delete':
                alignment1.extend(zip(self.source_text[source_begin:source_end],
                                      [self.gap_element]*(source_end-source_begin)))
            else:
                raise Exception("difflib returned invalid opcode", opcode, "in", self.source_text, self.target_text)
        assert source_end == len(self.source_text), \
            'alignment does not span full sequence "%s" - %d' % (self.source_text, source_end)
        assert target_end == len(self.target_text), \
            'alignment does not span full sequence "%s" - %d' % (self.target_text, target_end)

        # re-combine grapheme clusters by assigning
        # all combining codepoints to previous position,
        # leaving gap (but never combining with gap or non-letter):
        if not isinstance(self.source_text, list):
            alignment2 = []
            isnecessary = False
            for source_sym, target_sym in alignment1:
                if (source_sym != self.gap_element and
                    unicodedata.combining(source_sym) and
                    alignment2 and
                    alignment2[-1][0] != self.gap_element and
                    unicodedata.category(alignment2[-1][0][0])[0] == 'L'):
                    alignment2[-1][0] += source_sym
                    isnecessary = True
                    if target_sym == self.gap_element:
                        continue
                    elif (unicodedata.combining(target_sym) and
                          alignment2[-1][1] != self.gap_element and
                          unicodedata.category(alignment2[-1][1][0])[0] == 'L'):
                        alignment2[-1][1] += target_sym
                        continue
                    else:
                        source_sym = self.gap_element
                elif (target_sym != self.gap_element and
                      unicodedata.combining(target_sym) and
                      alignment2 and
                      alignment2[-1][1] != self.gap_element and
                      unicodedata.category(alignment2[-1][1][0])[0] == 'L'):
                    alignment2[-1][1] += target_sym
                    isnecessary = True
                    if source_sym == self.gap_element:
                        continue
                    else:
                        target_sym = self.gap_element
                alignment2.append([source_sym, target_sym])
            if isnecessary:
                alignment1 = list(map(tuple, alignment2))
        
        if self.confusion is not None:
            for pos, pair in enumerate(alignment1):
                # avoid gap in confusion, prefering multi-character entries instead
                # merge forward (since we know we always end with newline)
                if self.gap_element in pair:
                    continue
                def tplus(a, b):
                    return tuple(map(lambda x, y: (x or '') + (y or ''), a, b))
                while pos and self.gap_element in alignment1[pos - 1]:
                    pos -= 1
                    pair = tplus(alignment1[pos], pair)
                count = self.confusion.setdefault(pair, 0)
                self.confusion[pair] = count + 1
        
        return alignment1

    def get_confusion(self, limit=None):
        if self.confusion is None:
            raise Exception("aligner was not configured to count confusion")
        table = []
        class Confusion(object):
            def __init__(self, count, pair):
                self.count = count
                self.pair = pair
            def __repr__(self):
                return str((self.count, self.pair))
            def __lt__(self, other):
                return self.count > other.count
            def __le__(self, other):
                return self.count >= other.count
            def __eq__(self, other):
                return self.count == other.count
            def __ne__(self, other):
                return self.count != other.count
            def __gt__(self, other):
                return self.count < other.count
            def __ge__(self, other):
                return self.count <= other.count
        total = 0
        for pair, count in self.confusion.items():
            total += count
            if pair[0] == pair[1]:
                continue
            conf = Confusion(count, pair)
            length = len(table)
            idx = bisect.bisect_left(table, conf, hi=min(limit or length, length))
            if limit and idx >= limit:
                continue
            table.insert(idx, conf)
        if limit:
            table = table[:limit]
        return table, total
        
    def get_levenshtein_distance(self, source_text, target_text):
        """Align strings and calculate raw unweighted edit distance between its codepoints."""
        import editdistance
        dist = editdistance.eval(source_text, target_text)
        # not quite correct to use the largest sequence length:
        # this underestimates the alignment path length (which
        # we cannot get from the library)
        length = max(len(target_text), len(source_text))
        return dist/length if length else 0
    
    def get_adjusted_distance(self, source_text, target_text, normalization=None, gtlevel=1):
        """Normalize and align strings, recombining characters, and calculate unweighted edit distance.
        
        If ``normalization`` is a known Unicode canonicalization method,
        or equals ``'historic_latin'`` (denoting certain historic ligatures
        to be normalized when ``gtlevel<3``), then apply that transform
        to both ``source_text`` and ``target_text`` separately.
        
        Next, find the best alignment between their codepoints. Afterwards,
        recombine (sequences of) combining characters with preceding base letters.
        
        Finally, calculate the distance between these character strings.
        If ``normalization=='historic_latin'``, then treat certain semantically
        close pairs of characters as equal (not necessarily under NFC/NFKC).
        
        Return the arithmetic mean of the distances.
        """
        import unicodedata
        def normalize(seq):
            if normalization in ['NFC', 'NFKC']:
                if isinstance(seq, list):
                    return [unicodedata.normalize(normalization, tok) for tok in seq]
                else:
                    return unicodedata.normalize(normalization, seq)
            elif normalization == 'historic_latin':
                # multi-codepoint equivalences not involving combining characters:
                equivalences = { # keep only vocalic ligatures...
                    '': 'ſſ',
                    "\ueba7": 'ſſi',  # MUFI: LATIN SMALL LIGATURE LONG S LONG S I
                    '': 'ch',
                    '': 'ck',
                    '': 'll',
                    '': 'ſi',
                    '': 'ſt',
                    'ﬁ': 'fi',
                    'ﬀ': 'ff',
                    'ﬂ': 'fl',
                    'ﬃ': 'ffi',
                    '': 'ct',
                    '': 'tz',       # MUFI: LATIN SMALL LIGATURE TZ
                    '\uf532': 'as',  # eMOP: Latin small ligature as
                    '\uf533': 'is',  # eMOP: Latin small ligature is
                    '\uf534': 'us',  # eMOP: Latin small ligature us
                    '\uf535': 'Qu',  # eMOP: Latin ligature capital Q small u
                    'ĳ': 'ij',       # U+0133 LATIN SMALL LIGATURE IJ
                    '\uE8BF': 'q&',  # MUFI: LATIN SMALL LETTER Q LIGATED WITH FINAL ET  XXX How to replace this correctly?
                    '\uEBA5': 'ſp',  # MUFI: LATIN SMALL LIGATURE LONG S P
                    'ﬆ': 'st',      # U+FB06 LATIN SMALL LIGATURE ST
                    '\uF50E': 'q́' # U+F50E LATIN SMALL LETTER Q WITH ACUTE ACCENT
                } if gtlevel < 3 else {}
                equivalences = str.maketrans(equivalences)
                if isinstance(seq, list):
                    return [tok.translate(equivalences) for tok in seq]
                else:
                    return seq.translate(equivalences)
            else:
                return seq
        if normalization == 'historic_latin' and gtlevel == 1:
            equivalences = [
                # some of these are not even in NFKC:
                {"ä", "ä", "a\u0364"},
                {"ö", "ö", "o\u0364"},
                {"ü", "ü", "u\u0364"},
                {"Ä", "Ä", "A\u0364"},
                {"Ö", "Ö", "O\u0364"},
                {"Ü", "Ü", "U\u0364"},
                {"s", "ſ"},
                {"r", "ꝛ"},
                {"0", "⁰"},
                {"1", "¹"},
                {"2", "²"},
                {"3", "³"},
                {"4", "⁴"},
                {"5", "⁵"},
                {"6", "⁶"},
                {"7", "⁷"},
                {"8", "⁸"},
                {"9", "⁹", "ꝰ"},
                {"„", "»", "›", "〟"},
                {"“", "«", "‹", "〞"},
                {"'", "ʹ", "ʼ", "′", "‘", "’", "‛", "᾽"},
                {",", "‚"},
                {"-", "−", "—", "‐", "‑", "‒", "–", "⁃", "﹘", "―", "─"},
                {"‟", "〃", "”", "″"}, # ditto signs
                {"~", "∼", "˜", "῀", "⁓"},
                {"(", "⟨", "⁽"},
                {")", "⟩", "⁾"},
                {"/", "⧸", "⁄", "∕"},
                {"\\", "⧹", "∖", "⧵"}
            ]
        else:
            equivalences = []
        def equivalent(x, y):
            for equivalence in equivalences:
                if x in equivalence and y in equivalence:
                    return True
            return False

        self.set_seqs(normalize(source_text), normalize(target_text))
        alignment = self.get_best_alignment()
        
        dist = 0.0
        for source_sym, target_sym in alignment:
            #self.logger.debug('"%s"/"%s"', str(source_sym), str(target_sym))
            if source_sym == target_sym or equivalent(source_sym, target_sym):
                pass
            else:
                dist += 1.0
        # length = len(alignment) # normalized rate
        length = len(alignment)
            
        # FIXME: determine WER as well
        # idea: assign all non-spaces to previous position, leaving gap
        #       collapse gap-gap pairs, 
        
        return dist / length if length else 0
    
    @staticmethod
    def best_alignment(source_text, target_text):
        aligner = Alignment()
        aligner.set_seqs(source_text, target_text)
        return aligner.get_best_alignment()

class Edits():
    length = mean = varia = 0
    score = 0
    lines = 0
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    # numerically stable parallel/subsample aggregation algorithm by Chan et al. 1979:
    def update(self, length, mean, varia):
        if length < 1:
            return
        delta = mean - self.mean
        self.mean = (length * mean + self.length * self.mean) / (length + self.length)
        self.varia = (length * varia + self.length * self.varia +
                      delta ** 2 * length * self.length / (length + self.length))
        self.length += length
        self.varia /= self.length
        logging.getLogger('').debug('N=%d→%d µ=%.2f→%.2f σ²=%.2f→%.2f',
                                    length, self.length,
                                    mean, self.mean,
                                    varia, self.varia)
    
    def add(self, dist):
        self.update(1, dist, 0)
    
    def merge(self, edits):
        self.update(edits.length, edits.mean, edits.varia)
