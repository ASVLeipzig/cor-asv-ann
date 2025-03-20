from __future__ import absolute_import

import os
import multiprocessing as mp
from typing import Optional
from subprocess import run, PIPE
from unicodedata import category

import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd import Processor, Workspace, OcrdPageResult
from ocrd_models.ocrd_page import OcrdPage


@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann_mark(*args, **kwargs):
    return ocrd_cli_wrap_processor(MarkWords, *args, **kwargs)

class MarkWords(Processor):

    @property
    def executable(self):
        return 'ocrd-cor-asv-ann-mark'

    @property
    def metadata_filename(self) -> str:
        return os.path.join('wrapper', 'ocrd-tool.json')

    def process_workspace(self, workspace: Workspace) -> None:
        self.stats = mp.Manager().dict(total_candidates=0, total_nonmatches=0)
        super().process_workspace(workspace)
        total_nonmatches, total_candidates = self.stats['total_nonmatches'], self.stats['total_candidates']
        self.logger.info("marked %d unmatched words out of %d tokens (%d%%) overall",
                         total_nonmatches, total_candidates,
                         100 * total_nonmatches / total_candidates if total_candidates else 0)
        
    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Mark words that are not recognized by a spellchecker

        Open and deserialise PAGE input files, then iterate over the element hierarchy
        down to the word level. If there is no text or empty text, continue. Otherwise,
        normalize the text content by apply the character-wise `normalization`, and
        stripping any non-letters. Pass that string into `command`: if the output is
        not empty, then mark the word according to `format`.
        
        Produce new output files by serialising the resulting hierarchy.
        """
        pcgts = input_pcgts[0]
        command = self.parameter['command']
        format_ = self.parameter['format']
        n11n = self.parameter['normalization']
        def run_command(text):
            result = run(command, input=text, encoding="utf-8",
                         text=True, shell=True, capture_output=True)
            result.stdout = result.stdout.rstrip('\n')
            return result
        def asword(token):
            # apply normalization
            for nfrom, nto in n11n.items():
                token = token.replace(nfrom, nto)
            # strip punctuation etc
            result = ''
            for char in token:
                cat = category(char)[0]
                if cat in 'LM':
                    result += char
            return result
        def save(equiv):
            # save old state of hyphenated word to undo marking it as non-match
            if format_ == 'conf':
                oldattr = 'conf'
            else:
                oldattr = 'comments'
            oldvalue = getattr(equiv, oldattr)
            def fun():
                setattr(equiv, oldattr, oldvalue)
                self.logger.debug("Hyphenated word '%s' will be unmarked: '%s'", equiv.parent_object_.id, equiv.Unicode)
            return fun

        file_candidates = 0
        file_nonmatches = 0
        undo = None
        for region in pcgts.get_Page().get_AllRegions(classes=['Text']):
            for line in region.get_TextLine():
                words = line.get_Word()
                for word in words:
                    equiv = word.get_TextEquiv()
                    if not len(equiv):
                        self.logger.warning("Word '%s' contains no text results", word.id)
                        continue
                    text = equiv[0].Unicode
                    if not text:
                        self.logger.warning("Word '%s' contains empty text", word.id)
                        continue
                    text0 = asword(text)
                    if not text0:
                        self.logger.debug("Word '%s' has no letters: '%s'", word.id, text)
                        continue
                    result = run_command(text0)
                    file_candidates += 1
                    if result.returncode != 0:
                        self.logger.error("Word '%s' lookup failed (%d): '%s'", text0, result.returncode, result.stderr)
                    elif result.stdout:
                        assert text0 == result.stdout, (text0, result.stdout)
                        if undo and word is words[0]:
                            # try with dehyphenation
                            undo, text0 = undo
                            text = text0 + text
                            text0 = asword(text)
                            result = run_command(text0)
                            if result.returncode != 0:
                                self.logger.error("Word '%s' lookup failed (%d): '%s'", text0, result.returncode, result.stderr)
                            elif not result.stdout:
                                # undo previous word, skip this word
                                undo()
                                undo = None
                                file_nonmatches -= 1
                                continue
                        undo = None
                        if word is words[-1] and (text.endswith('-') or text.endswith('⸗')):
                            # save hyphenation candidate
                            undo = save(equiv[0]), text[:-1]
                        self.logger.debug("Word '%s' will be marked: '%s'", word.id, text0)
                        file_nonmatches += 1
                        if format_ == 'conf':
                            equiv[0].conf = 0.123
                        else:
                            equiv[0].comments = format_
        self.logger.info("marked %d unmatched words out of %d tokens (%d%%) on %s",
                         file_nonmatches, file_candidates,
                         100 * file_nonmatches / file_candidates if file_candidates else 0,
                         page_id)
        self.stats['total_candidates'] += file_candidates
        self.stats['total_nonmatches'] += file_nonmatches
        return OcrdPageResult(pcgts)
