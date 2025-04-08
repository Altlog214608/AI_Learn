import requests

url = 'http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList'
# params ={'serviceKey' : 'O9iepCIgT84Rh28Wc73CGiQesRVek87O9P30OpRy5TmYrIODe6MfSDSx7hYPbsiwGBmbwP28Y%2B%2FC6nNqbXbqhA%3D%3D', 'pageNo' : '1', 'numOfRows' : '10', 'dataType' : 'json', 'dataCd' : 'ASOS', 'dateCd' : 'DAY', 'startDt' : '20200220', 'endDt' : '20250407', 'stnIds' : '108',"_type": "json" }
params ={'serviceKey' : 'O9iepCIgT84Rh28Wc73CGiQesRVek87O9P30OpRy5TmYrIODe6MfSDSx7hYPbsiwGBmbwP28Y+/C6nNqbXbqhA==', 'pageNo' : '1', 'numOfRows' : '20', 'dataType' : 'json', 'dataCd' : 'ASOS', 'dateCd' : 'DAY', 'startDt' : '20250320', 'endDt' : '20250407', 'stnIds' : '108',"_type": "json" }

response = requests.get(url, params=params)

print(response.status_code)
print(response.text)

data = response.json()

items = data['response']['body']['items']['item']

for item in items:
    print("일자:", item['tm'])
    print("지점번호:", item['stnId'])
    print("일조시간(sumSsHr):", item.get('sumSsHr'))
    print("1시간 최대일사시각(hr1MaxIcsrHrmt):", item.get('hr1MaxIcsrHrmt'))
    print("1시간 최대일사량(hr1MaxIcsr):", item.get('hr1MaxIcsr'))
    print("총일사량(sumGsr):", item.get('sumGsr'))
    print("---")

