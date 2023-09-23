import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloads.download_client import DownloadClient
from config.config import config

if __name__ == '__main__':
    c = config()
    
    downloadClient = DownloadClient(c, num_threads=10)
    downloadClient.download()
    #downloadClient.fetch_pubmed_abstract_dates()