import click
import pprint


@click.group()
def cli():
    """Run recording software
    """


@click.command()
@click.argument("device_index", type=int)
@click.argument("channels", type=int, default=1)
def debug(device_index, channels):
    """Entrypoint into test application
    """
    from app import example_app, example_app_1
    example_app(device_index, channels)


@click.command()
def list_devices():
    import sounddevice as sd
    click.echo(sd.query_devices())


@click.command()
@click.argument("device_index", type=int)
def device_info(device_index):
    import sounddevice as sd
    click.echo(pprint.pformat(sd.query_devices()[device_index], indent=4))


cli.add_command(debug)
cli.add_command(list_devices)
cli.add_command(device_info)


if __name__ == "__main__":
    cli()
