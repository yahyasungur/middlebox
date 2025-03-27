from scapy.all import sniff, TCP

def extract_mss(packet):
    if TCP in packet and packet[TCP].options:
        for opt in packet[TCP].options:
            if opt[0] == 'MSS':
                print(f"Covert data extracted from MSS: {opt[1]}")

def start_tcp_listener():
    print("TCP listener started. Waiting for packets...")
    sniff(filter="tcp", prn=extract_mss)

if __name__ == "__main__":
    start_tcp_listener()