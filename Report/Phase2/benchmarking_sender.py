# benchmark_sender.py
from scapy.all import IP, TCP, send
import os
import time
import csv

def tcp_sender(covert_data, packet_count, interval):
    host = os.getenv('INSECURENET_HOST_IP')
    port = 8888

    if not host:
        print("INSECURENET_HOST_IP environment variable is not set.")
        return

    results = []

    try:
        for i in range(packet_count):
            ip = IP(dst=host)
            tcp = TCP(dport=port, options=[('MSS', int(covert_data))])
            packet = ip / tcp

            start_time = time.time()
            send(packet, verbose=False)
            end_time = time.time()

            print(f"Packet {i+1}/{packet_count} sent at {start_time}")
            results.append([i + 1, start_time, end_time])

            time.sleep(interval)

    except Exception as e:
        print(f"An error occurred: {e}")

    # Save results
    with open('send_times.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Packet No', 'Send Time', 'End Time'])
        writer.writerows(results)

    print("Benchmarking sender finished. Results saved to send_times.csv")

if __name__ == "__main__":
    covert_data = input("Enter covert data to embed in MSS field: ")
    packet_count = int(input("Number of packets to send: "))
    interval = float(input("Interval between packets (sec): "))
    tcp_sender(covert_data, packet_count, interval)
