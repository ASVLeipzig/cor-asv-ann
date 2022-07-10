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

TOOL_NAME = 'ocrd-cor-asv-ann-join'

@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann_join(*args, **kwargs):
    return ocrd_cli_wrap_processor(JoinLines, *args, **kwargs)

class JoinLines(Processor):
    
    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL_NAME]
        kwargs['version'] = OCRD_TOOL['version']
        super().__init__(*args, **kwargs)
        
    def process(self):
        """Join textlines of multiple file groups by concatenating their text results.
        
        Find files in all input file groups of the workspace for the same
        pageIds.
        
        Open and deserialise PAGE input files, then iterate over the element
        hierarchy down to the TextLine level. Concatenate the TextEquivs for
        all lines with the same TextLine IDs, optionally differentiating them
        by adding their original fileGrp name in @comments.
        
        Produce new output files by serialising the resulting hierarchy.
        """
        #assert_file_grp_cardinality(self.input_file_grp, >=1)
        assert_file_grp_cardinality(self.output_file_grp, 1)

        LOG = getLogger('processor.JoinLines')

        comments = self.parameter['add-filegrp-comments']
        
        ifgs = self.input_file_grp.split(",") # input file groups
        ninputs = len(ifgs)
        if ninputs < 2:
            raise Exception("need multiple input file groups to align")
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE) # input file tuples

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
                file_id2line[i] = {line.id: line for line in pcgts.get_Page().get_AllTextLines()}
                if not i:
                    # first input fileGrp becomes base for output fileGrp
                    output_pcgts = pcgts
                    self.add_metadata(output_pcgts)

            for line_id in file_id2line[0]:
                # get line objects among all input files for this line ID
                lines = [id2line.get(line_id, None) for id2line in file_id2line]
                line0 = lines[0]
                if comments:
                    for i, line in enumerate(lines):
                        if not line or not line.TextEquiv:
                            continue
                        for equiv in line.TextEquiv:
                            equiv.set_comments(ifgs[i])
                # get line texts among all input files for this line ID
                texts = itertools.from_iterable([line.TextEquiv for line in lines
                                                 # ignore missing lines and empty lines
                                                 if line and line.TextEquiv])
                # write back to line0
                line0.TextEquiv = texts

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

