import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from .transcode import ANNCorrection
from .evaluate import EvaluateLines

@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann_process(*args, **kwargs):
    return ocrd_cli_wrap_processor(ANNCorrection, *args, **kwargs)

@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann_evaluate(*args, **kwargs):
    return ocrd_cli_wrap_processor(EvaluateLines, *args, **kwargs)
