from __future__ import absolute_import

import os
import math
import json
import itertools
from typing import Optional

import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor
from ocrd import Processor, OcrdPageResult
from ocrd_models.ocrd_page import OcrdPage
from ocrd_utils import (
    MIMETYPE_PAGE
)


@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann_join(*args, **kwargs):
    return ocrd_cli_wrap_processor(JoinLines, *args, **kwargs)

class JoinLines(Processor):

    @property
    def executable(self):
        return 'ocrd-cor-asv-ann-join'
        
    @property
    def metadata_filename(self) -> str:
        return os.path.join('wrapper', 'ocrd-tool.json')

    def zip_input_files(self, **kwargs):
        kwargs['mimetype'] = MIMETYPE_PAGE
        return super().zip_input_files(**kwargs)

    def process_page_pcgts(self, *input_pcgts: Optional[OcrdPage], page_id: Optional[str] = None) -> OcrdPageResult:
        """Join textlines of multiple file groups by concatenating their text results.
        
        Find files in all input file groups of the workspace for the same
        pageIds.
        
        Open and deserialise PAGE input files, then iterate over the element
        hierarchy down to the TextLine level. Concatenate the TextEquivs for
        all lines with the same TextLine IDs (if `match-on=id`) or boundary
        points (if `match-on=coords`) or baseline points (if `match-on=baseline`).

        If `add-filegrp-comments` is true, then differentiate them by adding
        their original fileGrp name in @comments.
        If `add-filegrp-index` is true, then differentiate them by adding
        their original fileGrp index in @index.
        
        Produce new output files by serialising the resulting hierarchy.
        """
        comments = self.parameter['add-filegrp-comments']
        index = self.parameter['add-filegrp-index']
        match = self.parameter['match-on']
        def extract(line):
            if match == 'id':
                return line.id
            if match == 'baseline':
                if line.Baseline is None:
                    self.logger.error("cannot extract baseline from line '%s'", line.id)
                    return line.Coords.points
                return line.Baseline.points
            if match == 'coords':
                return line.Coords.points
        
        result = None
        master = 0
        ifgs = self.input_file_grp.split(',')
        file_id2line = [{} for _ in ifgs] # line ID dicts for this file
        for i, pcgts in enumerate(input_pcgts):
            if pcgts is None:
                # file/page was not found in this group
                continue
            file_id2line[i] = {extract(line): line
                               for line in pcgts.get_Page().get_AllTextLines()}
            if result is None:
                # first non-empty input fileGrp becomes base for output fileGrp
                result = OcrdPageResult(pcgts)
                master = i

        for line_id in file_id2line[master]:
            # get line objects across all input files for this line ID
            lines = [id2line.get(line_id, None) for id2line in file_id2line]
            line0 = lines[master]
            for i, line in enumerate(lines):
                if not line or not line.TextEquiv:
                    continue
                if comments:
                    for equiv in line.TextEquiv:
                        equiv.set_comments(ifgs[i])
                if index:
                    for equiv in line.TextEquiv:
                        equiv.set_index(i)
            # get line texts among all input files for this line ID
            texts = itertools.chain.from_iterable(
                [line.TextEquiv for line in lines
                 # ignore missing lines and empty lines
                 if line and line.TextEquiv])
            # write back to line0
            line0.TextEquiv = texts

        return result
