import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# 📁 데이터 불러오기
gdf_food = gpd.read_file(r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/식품점좌표/식품점노드.shp")
gdf_nodes = gpd.read_file(r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/shapefiles_package.gpkg", layer='nodes')

# 좌표계 일치
if gdf_food.crs != gdf_nodes.crs:
    gdf_food = gdf_food.to_crs(gdf_nodes.crs)

# ✏️ 1) node_id: 순번 인덱스가 아닌, 원래 고유키 NF_ID 사용
gdf_nodes['node_id'] = gdf_nodes['NF_ID']

# KDTree로 최근접 노드 계산
node_coords = np.column_stack((gdf_nodes.geometry.x, gdf_nodes.geometry.y))
food_coords = np.column_stack((gdf_food.geometry.x, gdf_food.geometry.y))
tree = cKDTree(node_coords)
distances, indices = tree.query(food_coords, k=1)

# ✏️ 2) indices → 실제 node_id 값으로 변환
gdf_food = gdf_food.reset_index(drop=True)
gdf_food['nearest_node_id']  = gdf_nodes.iloc[indices]['node_id'].values
gdf_food['distance_to_node'] = distances

# 병합: 이제 node_id 기준으로 올바르게 결합
merged_df = gdf_nodes.merge(
    gdf_food,
    left_on='node_id', right_on='nearest_node_id',
    how='left',
    suffixes=('_node', '_food')
)

# geometry 처리: 노드 geometry만 남기기
merged_df = merged_df.set_geometry('geometry_node').drop(columns=['geometry_food'])
merged_df = merged_df.rename_geometry('geometry')

# GeoDataFrame으로 변환 및 저장
merged_gdf = gpd.GeoDataFrame(merged_df, geometry='geometry', crs=gdf_nodes.crs)
output_path = r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/노드식품점결합/노드_식품점1N결합.gpkg"
merged_gdf.to_file(output_path, layer='노드_식품점결합', driver='GPKG')

print(f"✅ 저장 완료: {output_path}")

