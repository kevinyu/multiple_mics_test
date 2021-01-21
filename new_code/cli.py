import click
import pprint


@click.group()
def cli():
    """Run recording software
    """


@click.command()
@click.option("-c", "--config", type=click.Path(exists=True))
def gui(config):
    """Entrypoint into test application
    """
    from gui.main import run_app
    run_app(config)


@click.command()
@click.option("-c", "--config", type=click.Path(exists=True))
def listen(config):
    """Entrypoint into test application
    """
    from app import example_app
    example_app(config)


@click.command()
def list_devices():
    import sounddevice as sd
    click.echo(sd.query_devices())


@click.command()
@click.argument("device_index", type=int)
def device_info(device_index):
    import sounddevice as sd
    click.echo(pprint.pformat(sd.query_devices()[device_index], indent=4))


cli.add_command(gui)
cli.add_command(listen)
cli.add_command(list_devices)
cli.add_command(device_info)


if __name__ == "__main__":
    cli()
