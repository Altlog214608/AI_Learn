from netCDF4 import Dataset

# NetCDF 파일 열기
dataset = Dataset('123123.nc', 'r')

# 전체 변수 목록 보기
print(dataset.variables.keys())

# 특정 변수 데이터 가져오기
temperature = dataset.variables['gk2a_imager_projection'][:]  # 예시
print(temperature)
# print(len(temperature))

dataset.close()
