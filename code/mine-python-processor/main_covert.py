import asyncio
from nats.aio.client import Client as NATS
import os, random
from scapy.all import Ether, TCP

async def run():
    nc = NATS()

    nats_url = os.getenv("NATS_SURVEYOR_SERVERS", "nats://nats:4222")
    await nc.connect(nats_url)

    async def message_handler(msg):
        subject = msg.subject
        data = msg.data
        packet = Ether(data)
        if TCP in packet and packet[TCP].options:
            for opt in packet[TCP].options:
                if opt[0] == 'MSS':
                    covert_data = opt[1]
                    print(f"Processor extracted covert data: {covert_data}")
        # delay = random.expovariate(1 / 5e-6)
        # await asyncio.sleep(delay)
        
        if subject == "inpktsec":
            await nc.publish("outpktinsec", msg.data)
        else:
            await nc.publish("outpktsec", msg.data)

    await nc.subscribe("inpktsec", cb=message_handler)
    await nc.subscribe("inpktinsec", cb=message_handler)

    print("Subscribed to inpktsec and inpktinsec topics")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Disconnecting...")
        await nc.close()

if __name__ == '__main__':
    asyncio.run(run())