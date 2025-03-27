# benchmark_receiver.py
from scapy.all import sniff, TCP
import time
import csv
import signal
import sys

received_packets = []

def extract_mss(packet):
    if TCP in packet and packet[TCP].options:
        for opt in packet[TCP].options:
            if opt[0] == 'MSS':
                recv_time = time.time()
                print(f"Packet received at {recv_time}, MSS: {opt[1]}")
                received_packets.append([len(received_packets) + 1, recv_time, opt[1]])

def save_and_exit(sig, frame):
    print("\nStopping receiver. Saving data...")
    with open('recv_times.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Packet No', 'Receive Time', 'MSS'])
        writer.writerows(received_packets)
    print(f"Saved {len(received_packets)} packets to recv_times.csv")
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handler
    signal.signal(signal.SIGINT, save_and_exit)
    signal.signal(signal.SIGTERM, save_and_exit)

    print("TCP listener started. Press Ctrl+C to stop and save data.")
    sniff(filter="tcp", prn=extract_mss)
