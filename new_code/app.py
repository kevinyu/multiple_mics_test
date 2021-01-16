import asyncio
import logging

import numpy as np

from configure_logging import handler
from listen import Microphone
from visualize import Powerbar


logger = logging.getLogger(__name__)
logger.addHandler(handler)


async def main(device_index):
    pb = Powerbar(max_value=5000, channels=2)

    def echo(data):
        for i in range(data.shape[1]):
            pb.set_channel_value(i, np.max(np.abs(data[:, i])))
        pb.print()

    mic = Microphone(device_index=device_index)
    mic.SETUP.connect(lambda s: pb.set_channels(s["n_channels"]))
    mic.REC.connect(echo)
    mic.set_channels(2)
    mic.run()

    try:
        while True:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        mic.stop()


def example_app(device_index):
    asyncio.run(main(device_index))
