from scapy.all import PcapReader # https://stackoverflow.com/questions/68691090/python-scapy-error-no-libpcap-provider-available
import pdb

file_path = r'C:\Users\82090\Desktop\CS760_IoT_Traffic_Classification\DataSet\ArloQCamHTTPFlood_1.pcap'


if __name__ == '__main__':
    reader = PcapReader(file_path) # https://scapy.readthedocs.io/en/latest/api/scapy.utils.html?highlight=scapy+utils+pcapreader#scapy.utils.PcapReader
    pkts = reader.read_all()
    pdb.set_trace()

