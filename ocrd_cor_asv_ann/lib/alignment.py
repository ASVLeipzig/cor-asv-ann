# -*- coding: utf-8
import logging

class Alignment(object):
    def __init__(self, gap_element=0, logger=None):
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

        return alignment1
    
    def get_levenshtein_distance(self, source_text, target_text):
        # alignment for evaluation only...
        import editdistance
        dist = editdistance.eval(source_text, target_text)
        return dist, max(len(source_text), len(target_text))
    
    def get_adjusted_distance(self, source_text, target_text, normalization=None):
        import unicodedata
        def normalize(seq):
            if normalization in ['NFC', 'NFKC']:
                if isinstance(seq, list):
                    return [unicodedata.normalize(normalization, tok) for tok in seq]
                else:
                    return unicodedata.normalize(normalization, seq)
            else:
                return seq
        self.set_seqs(normalize(source_text), normalize(target_text))
        alignment = self.get_best_alignment()
        dist = 0 # distance
        
        umlauts = {u"ä": "a", u"ö": "o", u"ü": "u"} # for combination with U+0363 (not in NFKC)
        #umlauts = {}
        if normalization == 'historic_latin':
            equivalences = [
                # some of these are not even in NFKC:
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
        
        source_umlaut = ''
        target_umlaut = ''
        for source_sym, target_sym in alignment:
            #print(source_sym, target_sym)
            
            if source_sym == target_sym or equivalent(source_sym, target_sym):
                if source_umlaut: # previous source is umlaut non-error
                    source_umlaut = False # reset
                    dist += 1.0 # one full error (mismatch)
                elif target_umlaut: # previous target is umlaut non-error
                    target_umlaut = False # reset
                    dist += 1.0 # one full error (mismatch)
            else:
                if source_umlaut: # previous source is umlaut non-error
                    source_umlaut = False # reset
                    if (source_sym == self.gap_element and
                        target_sym == u"\u0364"): # diacritical combining e
                        dist += 1.0 # umlaut error (umlaut match)
                        #print('source umlaut match', a)
                    else:
                        dist += 2.0 # two full errors (mismatch)
                elif target_umlaut: # previous target is umlaut non-error
                    target_umlaut = False # reset
                    if (target_sym == self.gap_element and
                        source_sym == u"\u0364"): # diacritical combining e
                        dist += 1.0 # umlaut error (umlaut match)
                        #print('target umlaut match', a)
                    else:
                        dist += 2.0 # two full errors (mismatch)
                elif source_sym in umlauts and umlauts[source_sym] == target_sym:
                    source_umlaut = True # umlaut non-error
                elif target_sym in umlauts and umlauts[target_sym] == source_sym:
                    target_umlaut = True # umlaut non-error
                else:
                    dist += 1.0 # one full error (non-umlaut mismatch)
        if source_umlaut or target_umlaut: # previous umlaut error
            dist += 1.0 # one full error
        
        #length_reduction = max(source_text.count(u"\u0364"), target_text.count(u"\u0364"))
        return dist, max(len(source_text), len(target_text))
