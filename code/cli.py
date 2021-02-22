import click
import pprint


@click.group()
def cli():
    """Run recording software
    """


@click.command()
@click.option("-c", "--config", type=click.Path(exists=True), default=None)
def gui(config):
    """Entrypoint into test application
    """
    from gui.main import run_app
    run_app(config)


@click.command()
@click.option("-c", "--config", type=click.Path(exists=True))
@click.option("--save-on/--save-off", default=False)
def listen(config, save_on):
    """Entrypoint into test application
    """
    from app import headless_app
    headless_app(config, save_on)


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
