import asyncio
from nats.aio.client import Client as NATS
import os, random
from scapy.all import Ether, TCP

############## DL DETECTOR BEGIN ###################
import torch
import torch.nn as nn

class MSSClassifier(nn.Module):
    def __init__(self):
        super(MSSClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Load the trained model
detector_model = MSSClassifier()
detector_model.load_state_dict(torch.load('covertDetectorResources/mss_detector.pth'))
detector_model.eval()

def detect_covert(covert_data):
    mss_value = torch.tensor([[covert_data]], dtype=torch.float32)
    with torch.no_grad():
        prediction = detector_model(mss_value)
        predicted_label = (prediction > 0.5).item()

    if predicted_label == 1:
        return True
    else:
        return False
############## DL DETECTOR END ###################

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
                    mss_data = opt[1]
                    print(f"Processor extracted MSS data: {mss_data}")

                    # Use the DL model to detect covert channel activity
                    predicted_label = detect_covert(mss_data)

                    # If the model predicts covert channel activity, log it and print an alert
                    # If the model predicts covert channel activity, log it
                    with open("covert_detection_log.txt", "a") as log_file:
                        if predicted_label == True:
                            log_file.write(f"[ALERT] Covert channel activity detected! MSS={mss_data}\n")
                            print(f"[ALERT] Covert channel activity detected! MSS={mss_data}")
                        else:
                            log_file.write(f"[INFO] Normal traffic detected. MSS={mss_data}\n")
                    
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