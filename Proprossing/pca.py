import scapy
from scapy.utils import *
import pdb

file_path = r'C:\Users\82090\Desktop\CS760_IoT_Traffic_Classification\DataSet\ArloQCamHTTPFlood_1.pcap'

if __name__ == '__main__':
    pkts = rdpcap(file_path)
    pdb.set_trace()

