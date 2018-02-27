#Download the data
import urllib.request
import shutil
import zipfile

url = 'https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip'
file_name = 'data.zip'
print('Downloading dataset...')
with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
    shutil.copyfileobj(response, out_file)
print('Downloaded data.zip')
zip_ref = zipfile.ZipFile('./data.zip', 'r')
zip_ref.extractall('./data')
zip_ref.close()
print('Downloaded & Extracted')