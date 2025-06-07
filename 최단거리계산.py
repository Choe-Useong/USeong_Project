import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree
import networkx as nx
import os

# ────────────────────────────────────────
# 📁 1. 데이터 불러오기
# ────────────────────────────────────────
# 링크와 노드
gpkg_path = r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/shapefiles_package.gpkg"
gdf_links = gpd.read_file(gpkg_path, layer='links_with_adj')
gdf_nodes = gpd.read_file(r"C:\Users\admin\Desktop\식품사막프로젝트\1지도프로젝트\노드식품점결합\노드_식품점1N결합.gpkg")

# 인구격자 (강원 + 경기)
pop_paths = [
    r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/강원인구격자",
    r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/경기인구격자"
]
pop_gdfs = []

for path in pop_paths:
    gdf_pop = gpd.read_file(path)
    gdf_pop = gdf_pop[gdf_pop.columns[[0, 1, -1]]]  # id, 인구, geometry
    gdf_pop.columns = ['id', 'pop', 'geometry']
    gdf_pop['centroid'] = gdf_pop.geometry.centroid
    pop_gdfs.append(gdf_pop)

gdf_pop_all = pd.concat(pop_gdfs, ignore_index=True)
gdf_pop_all = gpd.GeoDataFrame(gdf_pop_all, geometry='centroid', crs=gdf_pop_all.crs)

# ────────────────────────────────────────
# 📍 2. 좌표계 통일
# ────────────────────────────────────────
if gdf_nodes.crs != gdf_pop_all.crs:
    gdf_pop_all = gdf_pop_all.to_crs(gdf_nodes.crs)
if gdf_links.crs != gdf_nodes.crs:
    gdf_links = gdf_links.to_crs(gdf_nodes.crs)

# ────────────────────────────────────────
# 📌 3. 인구 → 노드 매핑 (1:N → 합산)
# ────────────────────────────────────────
# 1. 인구 0 이상 필터링
gdf_pop_all['pop'] = pd.to_numeric(gdf_pop_all['pop'], errors='coerce').fillna(0)
gdf_pop_valid = gdf_pop_all[gdf_pop_all['pop'] > 0].copy()

# 2. 좌표 → 최근접 노드 찾기
pop_coords = np.array(list(zip(gdf_pop_valid.geometry.x, gdf_pop_valid.geometry.y)))
node_coords = np.array(list(zip(gdf_nodes.geometry.x, gdf_nodes.geometry.y)))

tree = cKDTree(node_coords)
_, indices = tree.query(pop_coords, k=1)
nearest_node_ids = gdf_nodes.iloc[indices].index

# 3. 전체에 NaN 초기화
gdf_pop_all['nearest_node'] = np.nan

# 4. 유효 인덱스에만 최근접 노드값 대입 (index 맞춰서!)
gdf_pop_all.loc[gdf_pop_valid.index, 'nearest_node'] = nearest_node_ids.values




tree = cKDTree(node_coords)
_, indices = tree.query(pop_coords, k=1)
nearest_node_ids = gdf_nodes.iloc[indices].index

# 노드 인구 합산
# 예: pop > 0인 행에만 최근접 노드 매핑
gdf_pop_all.loc[gdf_pop_valid.index, 'nearest_node'] = nearest_node_ids.values
pop_by_node = gdf_pop_all.groupby('nearest_node')['pop'].sum()
gdf_nodes['pop'] = gdf_nodes.index.map(pop_by_node).fillna(0)
gdf_nodes['pop'] = pd.to_numeric(gdf_nodes['pop'], errors='coerce').fillna(0)

# ────────────────────────────────────────
# 🛒 4. 대규모점포/슈퍼마켓 노드 필터링
# ────────────────────────────────────────
target_nodes = gdf_nodes[gdf_nodes['상권업종소'].isin(['대규모점포', '슈퍼마켓'])]

target_node_ids = target_nodes['NF_ID'].tolist()

# ────────────────────────────────────────
# 🕸️ 5. 링크로 그래프 생성
# ────────────────────────────────────────
from shapely.geometry import LineString



G = nx.DiGraph()  # 방향 있는 그래프

for idx, row in gdf_links.iterrows():
    try:
        start = row['BNODE_NFID']
        end = row['ENODE_NFID']
        dist = row.geometry.length
        osps = row['OSPS_SE']  # 일방 or 양방
        
        if dist >= 0:
            if osps == 'OWI001':  # 일방통행
                G.add_edge(start, end, weight=dist)
            else:  # 양방통행
                G.add_edge(start, end, weight=dist)
                G.add_edge(end, start, weight=dist)
    except:
        continue


# ────────────────────────────────────────
# 🧮 6. 최단 거리 계산
# ────────────────────────────────────────
distances = nx.multi_source_dijkstra_path_length(G, target_node_ids, weight='weight')

# 2) gdf_nodes의 NF_ID 순서에 맞춰 최단거리 리스트 생성
#    distances 사전에 없는 노드는 np.nan 처리
min_distances = [
    distances.get(node_id, np.nan)
    for node_id in gdf_nodes['NF_ID']
]


gdf_nodes['min_dist_to_store'] = min_distances

# pop과 거리 둘 다 숫자로 변환
gdf_nodes['pop'] = pd.to_numeric(gdf_nodes['pop'], errors='coerce')

# 곱셈 가능할 때만 계산, 그 외는 NaN
gdf_nodes['pop_dist'] = gdf_nodes['pop'] * gdf_nodes['min_dist_to_store']


# 예: gdf_nodes의 'pop_dist' 값을 인구격자에 할당
gdf_pop_all['pop_dist'] = gdf_pop_all['nearest_node'].map(gdf_nodes['pop_dist'])
gdf_pop_all['pop_dist'] = gdf_pop_all['pop_dist'].fillna(0)


# 예: 노드의 인구 값도 추가 가능
gdf_pop_all['node_pop'] = gdf_pop_all['nearest_node'].map(gdf_nodes['pop'])
gdf_pop_all['pop'] = pd.to_numeric(gdf_pop_all['pop'], errors='coerce')

# 로그 변환: log(1 + x) 형태로 0도 안정적으로 처리
gdf_pop_all['log_pop_dist'] = np.log1p(gdf_pop_all['pop_dist'])

gdf_pop_all['min_dist_to_store'] = gdf_pop_all['nearest_node'].map(gdf_nodes['min_dist_to_store']).fillna(0)

# ────────────────────────────────────────
# 💾 7. 저장
# ────────────────────────────────────────
output_path = r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/노드_최단거리_결과.gpkg"
gdf_nodes.to_file(output_path, layer="nodes", driver="GPKG")
print(f"✅ 저장 완료: {output_path}")


gdf_save = gdf_pop_all.drop(columns=['centroid'])

output_path = r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/인구격자_최단거리.gpkg"
gdf_save.to_file(output_path, layer="popgrid", driver="GPKG")

print(f"✅ 저장 완료: {output_path}")



# 대규모점포 또는 슈퍼마켓 필터링
target_nodes = gdf_nodes[gdf_nodes['상권업종소'].isin(['대규모점포', '슈퍼마켓'])]
output_path = r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/노드_대규모점포_슈퍼마켓.gpkg"
target_nodes.to_file(output_path, layer="target_nodes", driver="GPKG")









# 최근접 노드 ID 목록 (중복 제거)
nearest_node_ids = gdf_pop_all['nearest_node'].dropna().unique()

# gdf_nodes에서 해당 노드만 필터링
nearest_node_gdf = gdf_nodes[gdf_nodes.index.isin(nearest_node_ids)]

output_path = r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/노드_최근접인구.gpkg"
nearest_node_gdf.to_file(output_path, layer="nearest_nodes", driver="GPKG")








