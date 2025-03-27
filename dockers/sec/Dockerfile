FROM ubuntu:latest

#RUN apt update && apt install -y tshark iproute2 iptables dnsmasq jq netmask bsdmainutils conntrack iputils-ping dnsutils  net-tools 
RUN apt update && apt install -y tshark jq iputils-ping dnsutils  net-tools iperf tcpdump netcat-traditional curl  iproute2 ethtool iptables nftables

RUN echo 'net.ipv4.ip_forward=0' >> /etc/sysctl.conf && sysctl -p

# RUN apt update && apt install -y python3 python3-pip build-essential cmake libpcap-dev  vim 
# RUN pip install --upgrade pip && pip install scapy nats-py

RUN apt update && apt install -y python3 python3-pip python3-venv build-essential cmake libpcap-dev vim
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install scapy nats-py
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /code/sec