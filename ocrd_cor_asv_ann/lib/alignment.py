# -*- coding: utf-8
import logging
import bisect
import unicodedata

class Alignment():
    def __init__(self, gap_element=0, logger=None, confusion=False):
        self.confusion = dict() if confusion else None
        self.gap_element = gap_element
        self.logger = logger or logging.getLogger(__name__)
        
        ## difflib is optimised for visual comparisons (Ratcliff-Obershelp), not minimal distance (Levenshtein):
        from difflib import SequenceMatcher
        self.matcher = SequenceMatcher(isjunk=None, autojunk=False)
        
        self.source_text = []
        self.target_text = []
    
    def set_seqs(self, source_text, target_text):
        self.matcher.set_seqs(source_text, target_text)
        
        self.source_text = source_text
        self.target_text = target_text
        
    
    def is_bad(self):
        # self.matcher = difflib_matcher if len(source_text) > 4000 or len(target_text) > 4000 else editdistance_matcher
        # if self.matcher.distance() > 10 and self.matcher.distance() > len(self.source_text)-5:
        return bool(self.matcher.quick_ratio() < 0.5 and len(self.source_text) > 5)
    
    def get_best_alignment(self, eq=None):
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
