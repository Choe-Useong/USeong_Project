import geopandas as gpd
from scipy.spatial import cKDTree
import numpy as np
import pandas as pd

# 📁 데이터 불러오기
gdf_food = gpd.read_file(r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/식품점좌표/식품점노드.shp")
gdf_nodes = gpd.read_file(r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/shapefiles_package.gpkg", layer='nodes')

# 좌표계 일치
if gdf_food.crs != gdf_nodes.crs:
    gdf_food = gdf_food.to_crs(gdf_nodes.crs)

# 최근접 노드 찾기 (식품점 기준)
node_coords = np.array(list(zip(gdf_nodes.geometry.x, gdf_nodes.geometry.y)))
food_coords = np.array(list(zip(gdf_food.geometry.x, gdf_food.geometry.y)))
tree = cKDTree(node_coords)
distances, indices = tree.query(food_coords, k=1)

# 식품점별 최근접 노드 인덱스 및 거리 추가
gdf_food = gdf_food.reset_index(drop=True)
gdf_food['nearest_node_id'] = gdf_nodes.iloc[indices].index
gdf_food['distance_to_node'] = distances

# 노드 데이터에 'node_id' 추가
gdf_nodes = gdf_nodes.reset_index().rename(columns={'index': 'node_id'})

# 🏷️ 노드 기준으로 merge (노드 데이터에 식품점 정보 결합)
merged_df = gdf_nodes.merge(
    gdf_food,
    left_on='node_id', right_on='nearest_node_id',
    how='left',  # 노드 기준으로 결합 → 식품점 없는 노드도 포함
    suffixes=('_node', '_food')
)

# geometry 처리: 노드 geometry 유지
if 'geometry_food' in merged_df.columns:
    merged_df = merged_df.drop(columns=['geometry_food'])
if 'geometry_node' in merged_df.columns:
    merged_df = merged_df.rename(columns={'geometry_node': 'geometry'})

# GeoDataFrame 생성
merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=gdf_nodes.crs)

# 저장 (GeoPackage: 컬럼 제한 없음, 여러 geometry 타입 지원)
output_path = r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/노드식품점결합/노드_식품점1N결합.gpkg"
merged_gdf.to_file(output_path, layer='merged', driver='GPKG')

print(f"✅ GPKG로 저장 완료: {output_path}")





import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# 📁 데이터 불러오기
gdf_food = gpd.read_file(r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/식품점좌표/식품점좌표.shp")
gdf_nodes = gpd.read_file(r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/shapefiles_package.gpkg", layer='nodes')

# 좌표계 일치
if gdf_food.crs != gdf_nodes.crs:
    gdf_food = gdf_food.to_crs(gdf_nodes.crs)

# 인덱스 고정
gdf_nodes = gdf_nodes.reset_index(drop=True)
gdf_nodes['node_id'] = gdf_nodes.index

# KDTree로 최근접 노드 계산
node_coords = np.array(list(zip(gdf_nodes.geometry.x, gdf_nodes.geometry.y)))
food_coords = np.array(list(zip(gdf_food.geometry.x, gdf_food.geometry.y)))
tree = cKDTree(node_coords)
distances, indices = tree.query(food_coords, k=1)

# 결과 추가
gdf_food = gdf_food.reset_index(drop=True)
gdf_food['nearest_node_id'] = indices
gdf_food['distance_to_node'] = distances

# 병합
merged_df = gdf_nodes.merge(
    gdf_food,
    left_on='node_id', right_on='nearest_node_id',
    how='left',
    suffixes=('_node', '_food')
)

# geometry 처리
if 'geometry_food' in merged_df.columns:
    merged_df = merged_df.drop(columns=['geometry_food'])
if 'geometry_node' in merged_df.columns:
    merged_df = merged_df.rename(columns={'geometry_node': 'geometry'})

# GeoDataFrame 변환
merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=gdf_nodes.crs)

# 저장
output_path = "노드_식품점1N결합.gpkg"
merged_gdf.to_file(output_path, layer='노드_식품점결합', driver='GPKG')

print(f"✅ 저장 완료: {output_path}")








