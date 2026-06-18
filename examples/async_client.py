from biosiglive import AsyncTCPClient

import asyncio
import numpy as np


async def main():
    client = AsyncTCPClient("127.0.0.1", 12345)
    await client.connect()
    dt = 0.01
    t0 = 0

    while True:
        t = t0 + np.arange(30) * dt
        t0 += 30 * dt
        data = np.random.randn(10, 30).astype(np.float64)
        await client.send_array(data, sample_time=t)
        await asyncio.sleep(0.01)  # ~100 Hz


if __name__ == "__main__":
    asyncio.run(main())
