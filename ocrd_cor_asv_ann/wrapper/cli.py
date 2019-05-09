import click

from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from .transcode import ANNCorrection

@click.command()
@ocrd_cli_options
def ocrd_cor_asv_ann(*args, **kwargs):
    return ocrd_cli_wrap_processor(ANNCorrection, *args, **kwargs)
