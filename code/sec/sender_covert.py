from scapy.all import IP, TCP, send
import os

def tcp_sender(covert_data):
    host = os.getenv('INSECURENET_HOST_IP')
    port = 8888

    if not host:
        print("INSECURENET_HOST_IP environment variable is not set.")
        return

    try:
        while True:
            # Craft TCP packet with MSS option containing covert data
            ip = IP(dst=host)
            tcp = TCP(dport=port, options=[('MSS', int(covert_data))])
            packet = ip / tcp

            # Send the packet
            send(packet, verbose=False)
            print(f"Packet sent to {host}:{port} with MSS={covert_data}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    covert_data = input("Enter covert data to embed in MSS field: ")
    tcp_sender(covert_data)