# -*- coding: utf-8
import logging
from itertools import chain
from bisect import bisect_left, insort_left
import unicodedata
import uniseg.wordbreak

L1_HISTLAT_EQV = [
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

L2_HISTLAT_EQV = {
    # keep only vocalic ligatures...
    '\uf502': 'ch', # Latin small letter c ligated with latin small letter h, U+F502
    '\ueec4': 'ck', # Latin small ligature ck, U+EEC4
    'ﬅ': 'ſt',
    'ﬁ': 'fi',
    'ﬀ': 'ff',
    'ﬂ': 'fl',
    'ﬃ': 'ffi',
    '\uf4fc': 'ſk',
    '\ueedc': 'tz', # MUFI: LATIN SMALL LIGATURE TZ
    '\ueedc': 'tʒ', # PUA EEDC -> tʒ
    '\uf532': 'as', # eMOP: Latin small ligature as, U+f532
    '\uf533': 'is', # eMOP: Latin small ligature is, U+f533
    '\uf534': 'us', # eMOP: Latin small ligature us, U+f534
    '\uf535': 'Qu', # eMOP: Latin ligature capital Q small u, U+f535
    'ĳ': 'ij', # U+0133 LATIN SMALL LIGATURE IJ
    '\ue8bf': 'q&',  # MUFI: LATIN SMALL LETTER Q LIGATED WITH FINAL ET, U+E8BF
    '\ue8bf': 'qʒ', # PUA E8BF -> q; (or to qʒ, or to que, as you like)
    '\ueba5': 'ſp', # MUFI: LATIN SMALL LIGATURE LONG S P, U+EBA5
    'ﬆ': 'st', # U+FB06 LATIN SMALL LIGATURE ST
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
    '\ueada': 'ſt', # PUA EADA -> ſt
    '\ueba2': 'ſi', # PUA EBA2 -> ſi
    '\ueba3': 'ſl', # PUA EBA3 -> ſl
    '\ueba6': 'ſſ', # PUA EBA6 -> ſſ
    '\ueba7': 'ſſi', # MUFI: LATIN SMALL LIGATURE LONG S LONG S I, U+EBA7
    '\uf4ff': 'ſſt', # PUA F4FF -> ſſt
    '\uf52c': 'ſp', # PUA F52C -> ſp
    '\ueec5': 'ct', # PUA EEC5 -> ct
    '\ueecb': 'ft', # PUA EECB -> ft
    '\ue5d2': 'm̃', # PUA E5D2 -> m̃
    '\ue5dc': 'ñ', # PUA E5DC -> ñ
    '\ue665': 'p̃', # PUA E665 -> p + ...
    '\ue42c': 'aͤ', # PUA E42C -> a + U+0364, combining e above
    '\ue644': 'oͤ', # PUA E644 -> o + U+0364
    '\ue72b': 'uͤ', # PUA E72B -> u + U+0364
    '\ue72d': 'ů', # PUA E72D -> U+016F
    '\uebac': 'ß', # PUA EBAC -> ß (check for correct meaning)
    '\ue8b7': 'ß', # PUA E8B7 -> ß (proper replacement in some German printings)
    '\ue8b7': 'ſᷣ', # PUA E8B7 -> ſ with combining r rotunda (in some Latin printings)
    '\uf1a6': 'ꝰ', # PUA F1A6 -> U+A770, modifier letter us
    '\uf223': 'm', # PUA F223 -> m
    '\uf158': '⁊', # PUA F158 -> U+204A (Tironian et)
    '\uf159': 'ð', # PUA F159 -> eth, U+00F0
    '\uf160': ':', # PUA F160 -> :
    'q\uf02f': 'qͥ', # PUA F02F -> small letter i above (U+0365)
    't\uf1cc': 't᷑', # t + PUA F1CC -> t + combining ur above (U+1DD1)
    '\uf4f9': 'll', # PUA F4F9 -> ll
    # replace macron with tilde (easy to reach on keyboard and signalling abbreviations)
    'ā': 'ã',
    'ē': 'ẽ',
    'ī': 'ĩ',
    'ō': 'õ',
    'ū': 'ũ',
    'c̄': 'c̃',
    'q̄': 'q̃',
    'r̄': 'r̃',
    '\uf50e': 'q́', # U+F50E LATIN SMALL LETTER Q WITH ACUTE ACCENT
}

class Confusion():
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
        total = 0
        for pair, count in self.confusion.items():
            total += count
            if pair[0] == pair[1]:
                continue
            conf = Confusion(count, pair)
            length = len(table)
            idx = bisect_left(table, conf, hi=min(limit or length, length))
            if limit and idx >= limit:
                continue
            table.insert(idx, conf)
        if limit:
            table = table[:limit]
        return table, total
        
    @staticmethod
    def get_levenshtein_distance(source_text, target_text):
        """Align strings and calculate raw unweighted edit distance between its codepoints."""
        from rapidfuzz.distance import Levenshtein
        dist = Levenshtein.distance(source_text, target_text)
        # not quite correct to use the largest sequence length:
        # this underestimates the alignment path length (which
        # we cannot get from the library)
        length = max(len(target_text), len(source_text))
        return dist, length
    
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
        def normalize(seq):
            if isinstance(seq, list):
                return list(map(normalize, seq))
            if normalization in ['NFC', 'NFKC']:
                return unicodedata.normalize(normalization, seq)
            elif normalization == 'historic_latin':
                # multi-codepoint equivalences not involving combining characters:
                equivalences = L2_HISTLAT_EQV if gtlevel < 3 else {}
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
            equivalences = L1_HISTLAT_EQV
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

        if return_alignment:
            return dist, length, alignment
        return dist, length
    
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
    steps = 0
    hist1 = None
    hist2 = None
    worst = None

    class Example():
        mean = 0
        length = 0
        name = ''
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
        def cost(self):
            # for inverse order (worst first), return negative
            return - self.mean * self.length
        def __repr__(self):
            return (f"{self.name}: " if self.name else "") + \
                   (f"avg={self.mean} len={self.length}")
        def __lt__(self, other):
            return self.cost() < other.cost()
        def __le__(self, other):
            return self.cost() <= other.cost()
        def __eq__(self, other):
            return self.cost() == other.cost()
        def __ne__(self, other):
            return self.cost() != other.cost()
        def __gt__(self, other):
            return self.cost() > other.cost()
        def __ge__(self, other):
            return self.cost() >= other.cost()

    def __init__(self, logger=None, histogram=False):
        self.logger = logger or logging.getLogger(__name__)
        if histogram:
            self.hist1 = {'': 0}
            self.hist2 = {'': 0}
        else:
            self.hist1 = dict()
            self.hist2 = dict()
        self.worst = list()

    def __repr__(self):
        return 'N=%d µ=%.2f σ²=%.2f' % (
            self.length, self.mean, self.varia)

    def hist(self):
        keys = set(self.hist1.keys()).union(self.hist2.keys())
        bits = dict([(key, (self.hist1.get(key, 0), self.hist2.get(key, 0)))
                     for key in sorted(keys)])
        return bits
    
    # numerically stable parallel/subsample aggregation algorithm by Chan et al. 1979:
    def update(self, steps, length, mean, varia, hist1, hist2):
        if length < 1:
            return
        self.steps += steps
        delta = mean - self.mean
        self.mean = (length * mean + self.length * self.mean) / (length + self.length)
        self.varia = (length * varia + self.length * self.varia +
                      delta ** 2 * length * self.length / (length + self.length))
        self.length += length
        self.varia /= self.length
        self.logger.debug('N=%d→%d µ=%.2f→%.2f σ²=%.2f→%.2f',
                          length, self.length,
                          mean, self.mean,
                          varia, self.varia)
        for tok in hist1:
            self.hist1[tok] = hist1[tok] + self.hist1.setdefault(tok, 0)
        for tok in hist2:
            self.hist2[tok] = hist2[tok] + self.hist2.setdefault(tok, 0)
    
    def add(self, dist, length, seq1, seq2, name=None):
        hist1 = dict()
        hist2 = dict()
        if self.hist1:
            for tok in seq1:
                hist1[tok] = 1 + hist1.setdefault(tok, 0)
        if self.hist2:
            for tok in seq2:
                hist2[tok] = 1 + hist2.setdefault(tok, 0)
        self.update(1, length, dist / length if length else 0, 0, hist1, hist2)
        # aggregate outliers
        insort_left(self.worst, Edits.Example(mean=dist / length if length else 0, length=length, name=name))
        # reduce to worst 1% examples
        self.worst = self.worst[:max(int(self.steps * 0.01), 10)]
    
    def merge(self, edits, name_prefix=None):
        self.update(edits.steps, edits.length, edits.mean, edits.varia, edits.hist1, edits.hist2)
        if name_prefix:
            for example in edits.worst:
                example.name = name_prefix + example.name
        self.worst = sorted(chain(self.worst, edits.worst))[:max(int(self.steps * 0.01), 10)]

def _words(text):
    """segment a text into words"""
    # stolen from dinglehopper

    # Check if c is an unwanted character, i.e. whitespace, punctuation, or similar
    def unwanted(c):
        # See https://www.fileformat.info/info/unicode/category/index.htm
        # and https://unicodebook.readthedocs.io/unicode.html#categories
        unwanted_categories = "O", "M", "P", "Z", "S"
        unwanted_subcategories = "Cc", "Cf"

        subcat = unicodedata.category(c)
        cat = subcat[0]
        return cat in unwanted_categories or subcat in unwanted_subcategories

    # We follow Unicode Standard Annex #29 on Unicode Text Segmentation here: Split on
    # word boundaries using uniseg.wordbreak.words() and ignore all "words" that contain
    # only whitespace, punctation "or similar characters."
    for word in uniseg.wordbreak.words(text):
        if all(unwanted(c) for c in word):
            pass
        else:
            yield word

def splitwords(text):
    return list(_words(text))
