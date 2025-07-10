import pandas as pd
# 두 파일 각각 읽기
df_gangwon = pd.read_csv(r"C:\Users\admin\Desktop\식품사막프로젝트\상권데이터\소상공인시장진흥공단_상가(상권)정보_20250331\소상공인시장진흥공단_상가(상권)정보_강원_202503.csv")
df_gyeonggi = pd.read_csv(r"C:\Users\admin\Desktop\식품사막프로젝트\상권데이터\소상공인시장진흥공단_상가(상권)정보_20250331\소상공인시장진흥공단_상가(상권)정보_경기_202503.csv")

# 두 DataFrame을 수직(행 방향)으로 합치기
df = pd.concat([df_gangwon, df_gyeonggi], ignore_index=True)


import geopandas as gpd
from shapely.geometry import Point

df = df[['상호명', '상권업종소분류명', '표준산업분류명', '시군구명', '경도', '위도']]

# 2. 경도/위도 기반 Point 생성
geometry = [Point(xy) for xy in zip(df['경도'], df['위도'])]

# 3. GeoDataFrame으로 변환 (초기 좌표계: WGS84)
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# 4. EPSG:5174로 투영 좌표계 변환
gdf_5174 = gdf.to_crs("EPSG:5174")

# 5. EPSG:5174 좌표를 원래 df에 컬럼으로 추가
df['x_5174'] = gdf_5174.geometry.x
df['y_5174'] = gdf_5174.geometry.y



df['상권업종소분류명'].unique()
df['표준산업분류명'].unique()
#df = df[df['시군구명'].isin(['춘천시', '가평군', '인제군', '화천군', '홍천군'])]

a =df[df['상호명'].str.contains('장보고', na=False)]






bbb = df[df['상권업종소분류명'] == '슈퍼마켓']
bbb['표준산업분류명'].unique()
ccc = bbb[bbb['표준산업분류명'] == '그 외 기타 창작 및 예술관련 서비스업']





bbb = df[df['표준산업분류명'] == '슈퍼마켓']
bbb['상권업종소분류명'].unique()
ccc = bbb[bbb['상권업종소분류명'] == '편의점']





df2 = pd.read_excel(r"C:\Users\admin\Desktop\식품사막프로젝트\상권데이터\08_25_01_P_대규모점포.xlsx")
df2 = df2.dropna(subset=['소재지면적'])



사업장명목록 = df2['사업장명'].dropna().unique().tolist()

삭제대상 = df[df['상호명'].apply(lambda x: any(name in x for name in 사업장명목록) if pd.notnull(x) else False)]
df = df[~df['상호명'].apply(lambda x: any(name in x for name in 사업장명목록) if pd.notnull(x) else False)]




# 📌 4. df2 정리 후 구조 맞추기
df2_정리 = pd.DataFrame({
    '상호명': df2['사업장명'],
    '상권업종소분류명': '대규모점포',
    '표준산업분류명': '대규모점포',
    '시군구명': '춘천시',
    'x_5174': df2['좌표정보X(EPSG5174)'],
    'y_5174': df2['좌표정보Y(EPSG5174)']
})

# 📌 5. 두 데이터프레임 병합
df_통합 = pd.concat([df, df2_정리], ignore_index=True)




# 유지할 상권업종소분류명 리스트
유지_업종 = [
    '채소/과일 소매업',
    '슈퍼마켓',
    '편의점',
    '곡물/곡분 소매업',
    '수산물 소매업',
    '정육점',
    '대규모점포'
]

# 조건에 맞는 행만 필터링
df_통합 = df_통합[df_통합['상권업종소분류명'].isin(유지_업종)]




제외_산업분류_편의점 = [
    '한식 일반 음식점업',
    '기타 음ㆍ식료품 위주 종합 소매업',
    '빵류; 과자류 및 당류 소매업',
    '섬유; 의복; 신발 및 가죽제품 중개업'
]

df_통합 = df_통합[~(
    (df_통합['상권업종소분류명'] == '편의점') & 
    (df_통합['표준산업분류명'].isin(제외_산업분류_편의점))
)]




# 🧹 1. '슈퍼마켓' 중 제거할 표준산업분류명
제외_분류 = [
    '음료 소매업',
    '기타 일반 및 생활 숙박시설 운영업',
    '의복 액세서리 및 모조 장신구 소매업',
    '빵류; 과자류 및 당류 소매업',
    '체형 등 기타 신체 관리 서비스업',
    '컴퓨터 프로그래밍 서비스업',
    '그 외 기타 창작 및 예술관련 서비스업'
]

# 필터링: 슈퍼마켓 + 제외분류 조합 제거
df_통합 = df_통합[~(
    (df_통합['상권업종소분류명'] == '슈퍼마켓') &
    (df_통합['표준산업분류명'].isin(제외_분류))
)]

# 🏷️ 2. 특정 산업분류명을 가진 슈퍼마켓을 '소규모슈퍼마켓'으로 변경
변경_조건 = (
    (df_통합['상권업종소분류명'] == '슈퍼마켓') &
    (df_통합['표준산업분류명'] == '기타 음ㆍ식료품 위주 종합 소매업')
)
df_통합.loc[변경_조건, '상권업종소분류명'] = '소규모슈퍼마켓'
df_통합 = df_통합.drop(columns=['경도', '위도'])



from shapely.geometry import Point

# 1. Point geometry 생성
df_통합['geometry'] = [Point(xy) for xy in zip(df_통합['x_5174'], df_통합['y_5174'])]

# 2. GeoDataFrame으로 변환 (좌표계: EPSG:5174)
gdf_노드 = gpd.GeoDataFrame(df_통합, geometry='geometry', crs='EPSG:5174')

# 3. Shapefile로 저장 (폴더 및 파일명 지정)
gdf_노드.to_file(r"C:\Users\admin\Desktop\식품사막프로젝트\1지도프로젝트\식품점좌표\식품점노드.shp", encoding='cp949')


import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

# 1. 파일 불러오기
gdf_nodes = gpd.read_file(r"C:/Users/admin/Desktop/춘천노드/춘천노드.shp")
gdf_gy = gpd.read_file(r"C:\Users\admin\Documents\경로당\경로당.shp")
gdf_food = gpd.read_file(r"C:\Users\admin\Desktop\식품노드\식품점노드.shp")

# 2. 좌표계 통일
gdf_nodes = gdf_nodes.to_crs("EPSG:5186")
gdf_gy = gdf_gy.to_crs("EPSG:5186")
gdf_food = gdf_food.to_crs("EPSG:5186")

# 3. KDTree 구성
node_coords = np.array([(geom.x, geom.y) for geom in gdf_nodes.geometry])
tree = cKDTree(node_coords)

# 4. 경로당 snap: NF_ID와 시설명 추출
gy_snap_info = []
for i, pt in enumerate(gdf_gy.geometry):
    _, idx = tree.query([pt.x, pt.y])
    nf_id = gdf_nodes.iloc[idx]['NF_ID']
    name = gdf_gy.iloc[i]['시설명']  # 시설명 컬럼명 정확히 확인 필요
    gy_snap_info.append((nf_id, name))

# 5. 식품점 snap: NF_ID와 업종명 추출
food_snap_info = []
for i, pt in enumerate(gdf_food.geometry):
    _, idx = tree.query([pt.x, pt.y])
    nf_id = gdf_nodes.iloc[idx]['NF_ID']
    category = gdf_food.iloc[i]['상권업종소']
    food_snap_info.append((nf_id, category))

# 6. NF_ID 기준 집계 (중복 노드는 하나만 할당)
gy_df = gpd.GeoDataFrame(gy_snap_info, columns=['NF_ID', '경로당명']).drop_duplicates('NF_ID')
food_df = gpd.GeoDataFrame(food_snap_info, columns=['NF_ID', '식품점종류']).drop_duplicates('NF_ID')

# 7. 춘천노드에 merge
gdf_nodes = gdf_nodes.merge(gy_df, on='NF_ID', how='left')
gdf_nodes = gdf_nodes.merge(food_df, on='NF_ID', how='left')

# 8. 여부 컬럼 생성
gdf_nodes['경로당여부'] = gdf_nodes['경로당명'].notna().astype(int)
gdf_nodes['식품점여부'] = gdf_nodes['식품점종류'].notna().astype(int)

# 9. 저장
output_path = r"C:/Users/admin/Desktop/춘천노드_경로당식품점종류포함.shp"
gdf_nodes.to_file(output_path, encoding='cp949')

