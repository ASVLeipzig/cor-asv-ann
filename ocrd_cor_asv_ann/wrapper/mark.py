from __future__ import absolute_import

import os
from subprocess import run, PIPE
from unicodedata import category
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

TOOL = 'ocrd-cor-asv-ann-mark'

@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann_mark(*args, **kwargs):
    return ocrd_cli_wrap_processor(MarkWords, *args, **kwargs)

class MarkWords(Processor):
    
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)
        
    def process(self):
        """Mark words that are not recognized by a spellchecker

        Open and deserialise PAGE input files, then iterate over the element hierarchy
        down to the word level. If there is no text or empty text, continue. Otherwise,
        normalize the text content by apply the character-wise `normalization`, and
        stripping any non-letters. Pass that string into `command`: if the output is
        not empty, then mark the word according to `format`.
        
        Produce new output files by serialising the resulting hierarchy.
        """
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 1)
        LOG = getLogger('processor.MarkWords')
        command = self.parameter['command']
        format_ = self.parameter['format']
        n11n = self.parameter['normalization']
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
        for n, input_file in enumerate(self.input_files):
            LOG.info("INPUT FILE %i / %s", n, input_file.pageId or input_file.ID)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            page_id = input_file.pageId or input_file.ID
            file_id = make_file_id(input_file, self.output_file_grp)
            pcgts.set_pcGtsId(file_id)
            self.add_metadata(pcgts)
            for region in pcgts.get_Page().get_AllRegions(classes=['Text']):
                for line in region.get_TextLine():
                    for word in line.get_Word():
                        equiv = word.get_TextEquiv()
                        if not len(equiv):
                            LOG.warning("Word '%s' contains no text results", word.id)
                            continue
                        text = equiv[0].Unicode
                        if not text:
                            LOG.warning("Word '%s' contains empty text", word.id)
                            continue
                        text0 = asword(text)
                        if not text0:
                            LOG.debug("Word '%s' has no letters: '%s'", word.id, text)
                            continue
                        text = text0
                        result = run(command, input=text, encoding="utf-8", universal_newlines=True,
                                     shell=True, stdout=PIPE, stderr=PIPE)
                        if result.returncode != 0:
                            LOG.error("Word '%s' lookup failed (%d): '%s'", word.id, result.returncode, result.stderr)
                            continue
                        result.stdout = result.stdout.rstrip('\n')
                        if result.stdout:
                            assert text == result.stdout, (text, result.stdout)
                            LOG.debug("Word '%s' will be marked: '%s'", word.id, text)
                            if format_ == 'conf':
                                equiv[0].conf = 0.123
                            else:
                                equiv[0].comments = format_
            file_path = os.path.join(self.output_file_grp, file_id + '.xml')
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype=MIMETYPE_PAGE,
                content=to_xml(pcgts))
