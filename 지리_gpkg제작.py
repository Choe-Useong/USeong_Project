import geopandas as gpd
import os
import pandas as pd

# 🔍 파일 경로 리스트
file_paths = [
    r"C:\Users\admin\Desktop\식품사막프로젝트\국가기본도_차도링크\TN_RODWAY_LINK.shp",
    r"C:\Users\admin\Desktop\식품사막프로젝트\국가기본도_차도링크\TN_RODWAY_LINK(1).shp",
    r"C:\Users\admin\Desktop\식품사막프로젝트\국가기본도 차도노드\TN_RODWAY_NODE.shp",
    r"C:\Users\admin\Desktop\식품사막프로젝트\국가기본도 차도노드\TN_RODWAY_NODE(1).shp"
]

gpkg_path = r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/shapefiles_package.gpkg"

# 🔍 노드 & 링크 병합 및 저장
merge_groups = {
    "nodes": [path for path in file_paths if 'NODE' in os.path.basename(path).upper()],
    "links": [path for path in file_paths if 'LINK' in os.path.basename(path).upper()]
}

for group_name, paths in merge_groups.items():
    gdf_list = [gpd.read_file(path) for path in paths]
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)
    merged_gdf.to_file(gpkg_path, layer=group_name, driver="GPKG")
    print(f"📦 '{group_name}' 병합 완료 및 저장 ({len(merged_gdf)} 행)")

print(f"\n🎉 병합 저장 완료: {gpkg_path}")

# 노드 및 링크 레이어 불러오기
gdf_nodes = gpd.read_file(gpkg_path, layer='nodes')
gdf_links = gpd.read_file(gpkg_path, layer='links')


from shapely.geometry import LineString

# NF_ID → geometry 매핑
id_to_geom = dict(zip(gdf_nodes['NF_ID'], gdf_nodes.geometry))

# NF_ID → geometry 매핑
id_to_geom = dict(zip(gdf_nodes['NF_ID'], gdf_nodes.geometry))

# ✅ NF_ID → 법정동코드(앞 5자리) 매핑
id_to_leglcd = {}
for _, row in gdf_nodes.iterrows():
    code = None
    if pd.notnull(row.get("LEGLCD_SE1")):
        code = row["LEGLCD_SE1"]
    elif pd.notnull(row.get("LEGLCD_SE2")):
        code = row["LEGLCD_SE2"]
    id_to_leglcd[row['NF_ID']] = code

# 기존 링크의 컬럼 구조 확보
link_columns = gdf_links.columns

# 가상 링크 생성
virtual_links = []
for _, row in gdf_nodes.iterrows():
    src = row['NF_ID']
    tgt = row['NBND_NFID']

    if pd.notnull(tgt) and tgt in id_to_geom:
        try:
            geom = LineString([id_to_geom[src], id_to_geom[tgt]])
            leglcd = id_to_leglcd.get(src) or id_to_leglcd.get(tgt)
            virtual_links.append({
                'NF_ID': f'{src}_to_{tgt}',
                'BNODE_NFID': src,
                'ENODE_NFID': tgt,
                'geometry': geom,
                'RDLINK_SE': 'ADJ',
                'OSPS_SE': 'OWI002',
                'ROAD_NO': 'adj_link',
                'TFCEQP_SE': '00',
                'LEGLCD_SE': leglcd  # ✅ 여기 추가
            })
        except Exception as e:
            print(f"❌ 가상도로 생성 실패: {src} → {tgt}, 이유: {e}")


# GeoDataFrame으로 변환
gdf_virtual_links = gpd.GeoDataFrame(virtual_links, crs=gdf_nodes.crs)

# 누락된 필드 채우기 (gdf_links 구조에 맞춤)
for col in link_columns:
    if col not in gdf_virtual_links.columns:
        gdf_virtual_links[col] = None
gdf_virtual_links = gdf_virtual_links[link_columns]

# 병합
gdf_links_with_virtual = pd.concat([gdf_links, gdf_virtual_links], ignore_index=True)

# 💾 저장
gdf_links_with_virtual.to_file(gpkg_path, layer='links_with_adj', driver='GPKG')
print(f"💾 'links_with_adj' 레이어 저장 완료 ({len(gdf_links_with_virtual)} 행)")

print(f"✅ nodes 레이어 불러오기 완료 ({len(gdf_nodes)} 행)")
print(f"✅ links 레이어 불러오기 완료 ({len(gdf_links)} 행)")


