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
                if eq and eq(*pair):
                    continue
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
    
    def get_adjusted_distance(self, source_text, target_text, normalization=None, gtlevel=1, return_alignment=False):
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
            if isinstance(seq, list):
                return list(map(normalize, seq))
            if normalization in ['NFC', 'NFKC']:
                return unicodedata.normalize(normalization, seq)
            elif normalization == 'historic_latin':
                # multi-codepoint equivalences not involving combining characters:
                equivalences = { # keep only vocalic ligatures...
                    '': 'ſſ',
                    "": 'ſſi',  # MUFI: LATIN SMALL LIGATURE LONG S LONG S I, U+EBA7
                    '': 'ch', # Latin small letter c ligated with latin small letter h, U+F502
                    '': 'ck', # Latin small ligature ck, U+EEC4
                    'ﬅ': 'ſt',
                    'ﬁ': 'fi',
                    'ﬀ': 'ff',
                    'ﬂ': 'fl',
                    'ﬃ': 'ffi',
                    '': 'ſk',
                    '': 'tz',       # MUFI: LATIN SMALL LIGATURE TZ
                    '': 'as',  # eMOP: Latin small ligature as, U+f532
                    '': 'is',  # eMOP: Latin small ligature is, U+f533
                    '': 'us',  # eMOP: Latin small ligature us, U+f534
                    '': 'Qu',  # eMOP: Latin ligature capital Q small u, U+f535
                    'ĳ': 'ij',       # U+0133 LATIN SMALL LIGATURE IJ
                    '': 'q&',  # MUFI: LATIN SMALL LETTER Q LIGATED WITH FINAL ET, U+E8BF
                    '': 'ſp',  # MUFI: LATIN SMALL LIGATURE LONG S P, U+EBA5
                    'ﬆ': 'st',      # U+FB06 LATIN SMALL LIGATURE ST
                    'q̈': 'qᷓ', # replace combining diaeresis with flattened a above (abbrev.: quam)
                    'c̈': 'cᷓ', # (abbrev.: cetera)
                    'ḡ': 'gᷓ', # U+1E21 -> g + U1DD3 (ang- or gna-)
                    # use combining r rotunda (U+1DE3, ᷣ) instead of combining ogonek above (U+1DCE, ᷎)
                    # or combining hook above (U+0309, ̉); adapt to all your combinations
                    'v̉': 'vᷣ', # combining hook above -> comb. r rotunda, U+1DE3
                    'v᷎': 'vᷣ', # combining ogonek above -> comb. r rotunda, U+1DE3
                    'b᷎': 'bᷣ', # combining ogonek above -> comb. r rotunda, U+1DE3
                    'p᷎': 'pᷣ', # combining ogonek above -> comb. r rotunda, U+1DE3
                    # exception: d + comb. r rotunda is hardly visible on screen with most fonts, so use eth instead for the d + something
                    'd̉': 'ð', # d+comb. hook > eth, U+00F0 (CTRL-d on Linux keyboard)
                    'ꝟ': 'vᷣ', # U+A75F -> v with comb. r rotunda, U+1DE3
                    'tᷣ': 't᷑', # comb. r above -> combining ur above, U+1DD1 (in Latin passives such as dat᷑ = datur)
                    # replace font dependent PUA code points with accepted Unicodes
                    '': 'ſt', # PUA EADA -> ſt
                    '': 'ſi', # PUA EBA2 -> ſi
                    '': 'ſl', # PUA EBA3 -> ſl
                    '': 'ſſ', # PUA EBA6 -> ſſ
                    '': 'ſſi', # PUA EBA7 -> ſſi
                    '': 'ſſt', # PUA F4FF -> ſſt
                    '': 'ſp', # PUA F52C -> ſp
                    '': 'ct', # PUA EEC5 -> ct
                    '': 'ft', # PUA EECB -> ft
                    '': 'tʒ', # PUA EEDC -> tʒ
                    '': 'm̃', # PUA E5D2 -> m̃
                    '': 'ñ', # PUA E5DC -> ñ
                    '': 'p̃', # PUA E665 -> p + ...
                    '': 'qʒ', # PUA E8BF -> q; (or to qʒ, or to que, as you like)
                    '': 'aͤ', # PUA E42C -> a + U+0364, combining e above
                    '': 'oͤ', # PUA E644 -> o + U+0364
                    '': 'uͤ', # PUA E72B -> u + U+0364
                    '': 'ů', # PUA E72D -> U+016F
                    '': 'ß', # PUA EBAC -> ß (check for correct meaning)
                    '': 'ß', # PUA E8B7 -> ß (proper replacement in some German printings)
                    #'': 'ſᷣ', # PUA E8B7 -> ſ with combining r rotunda (in some Latin printings)
                    '': 'ꝰ', # PUA F1A6 -> U+A770, modifier letter us
                    '': 'm', # PUA F223 -> m
                    '': '⁊', # PUA F158 -> U+204A (Tironian et)
                    '': 'ð', # PUA F159 -> eth, U+00F0
                    '': ':', # PUA F160 -> :
                    'q': 'qͥ', # PUA F02F -> small letter i above (U+0365)
                    't': 't᷑', # t + PUA F1CC -> t + combining ur above (U+1DD1)
                    '': 'll', # PUA F4F9 -> ll
                    # replace macron with tilde (easy to reach on keyboard and signalling abbreviations)
                    'ā': 'ã',
                    'ē': 'ẽ',
                    'ī': 'ĩ',
                    'ō': 'õ',
                    'ū': 'ũ',
                    'c̄': 'c̃',
                    'q̄': 'q̃',
                    'r̄': 'r̃',
                    '': 'q́' # U+F50E LATIN SMALL LETTER Q WITH ACUTE ACCENT
                } if gtlevel < 3 else {}
                equivtab = dict()
                for key in list(equivalences):
                    if len(key) == 1:
                        equivtab[key] = equivalences.pop(key)
                equivtab = str.maketrans(equivtab)
                for key in equivalences:
                    seq = seq.replace(key, equivalences[key])
                return seq.translate(equivtab)
            else:
                return seq
        if normalization == 'historic_latin' and gtlevel == 1:
            equivalences = [
                # some of these are not even in NFKC:
                {"ä", "ä", "a\u0364"}, # a umlaut: precomposed, decomposed, combinine e
                {"ö", "ö", "o\u0364"}, # o umlaut: precomposed, decomposed, combinine e
                {"ü", "ü", "u\u0364"}, # u umlaut: precomposed, decomposed, combinine e
                {"Ä", "Ä", "A\u0364"}, # A umlaut: precomposed, decomposed, combinine e
                {"Ö", "Ö", "O\u0364"}, # O umlaut: precomposed, decomposed, combinine e
                {"Ü", "Ü", "U\u0364"}, # U umlaut: precomposed, decomposed, combinine e
                #{"I", "J"} # most Fraktur fonts have only a single glyph for I and J
                {"s", "ſ"}, # LATIN SMALL LETTER LONG S, U+017F
                {"r", "ꝛ"}, # LATIN SMALL LETTER R ROTUNDA, U+A75B
                {"z", "ʒ"}, # LATIN SMALL LETTER EZH/YOGH, U+0292
                {"Z", "Ʒ"}, # LATIN CAPITAL LETTER EZH/YOGH, U+01B7
                {"n", "ƞ"}, # LATIN SMALL LETTER N WITH LONG RIGHT LEG, U+019E
                {"μ", "µ"}, # Greek vs math mu
                {"π", "𝛑", "𝜋", "𝝅", "𝝿", "𝞹"}, # Greek vs math pi
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
                {"„", "»", "›", "〟"}, # opening double quotes
                {"“", "«", "‹", "〞"}, # closing double quotes
                {"'", "ʹ", "ʼ", "′", "‘", "’", "‛", "᾽", "`"}, # single quotes
                {",", "‚"}, # SINGLE LOW-9 QUOTATION MARK, U+201A
                {"-", "−", "—", "‐", "‑", "‒", "–", "⁃", "﹘", "―", "─", "⸗"},
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
            if isinstance(x, list):
                return len(x) == len(y) and all(
                    (equivalent(xc, yc) for xc, yc in zip(x, y)))
            if x == y:
                return True
            for equivalence in equivalences:
                if x in equivalence and y in equivalence:
                    return True
            return False

        self.set_seqs(normalize(source_text), normalize(target_text))
        alignment = self.get_best_alignment(eq=equivalent)
        
        dist = 0.0
        for source_sym, target_sym in alignment:
            #self.logger.debug('"%s"/"%s"', str(source_sym), str(target_sym))
            if source_sym == target_sym or equivalent(source_sym, target_sym):
                pass
            else:
                dist += 1.0
        length = len(alignment)

        rate = dist / length if length else 0
        if return_alignment:
            return rate, alignment
        return rate
    
    @staticmethod
    def best_alignment(source_text, target_text, with_confusion=False):
        aligner = Alignment(confusion=with_confusion)
        aligner.set_seqs(source_text, target_text)
        if with_confusion:
            return aligner.get_best_alignment(), aligner.get_confusion()
        return aligner.get_best_alignment()

class Edits():
    length = mean = varia = 0
    score = 0
    lines = 0
    hist1 = None
    hist2 = None
    def __init__(self, logger=None, histogram=False):
        self.logger = logger or logging.getLogger(__name__)
        if histogram:
            self.hist1 = {'': 0}
            self.hist2 = {'': 0}
        else:
            self.hist1 = dict()
            self.hist2 = dict()

    def hist(self):
        keys = set(self.hist1.keys()).union(self.hist2.keys())
        bits = dict([(key, (self.hist1.get(key, 0), self.hist2.get(key, 0)))
                     for key in sorted(keys)])
        return bits
    
    # numerically stable parallel/subsample aggregation algorithm by Chan et al. 1979:
    def update(self, length, mean, varia, hist1, hist2):
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
        for tok in hist1:
            self.hist1[tok] = hist1[tok] + self.hist1.setdefault(tok, 0)
        for tok in hist2:
            self.hist2[tok] = hist2[tok] + self.hist2.setdefault(tok, 0)
    
    def add(self, dist, seq1, seq2):
        hist1 = dict()
        hist2 = dict()
        if self.hist1:
            for tok in seq1:
                hist1[tok] = 1 + hist1.setdefault(tok, 0)
        if self.hist2:
            for tok in seq2:
                hist2[tok] = 1 + hist2.setdefault(tok, 0)
        self.update(1, dist, 0, hist1, hist2)
    
    def merge(self, edits):
        self.update(edits.length, edits.mean, edits.varia, edits.hist1, edits.hist2)
