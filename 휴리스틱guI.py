import pandas as pd
import numpy as np
import geopandas as gpd
import requests
from shapely.geometry import Point
from scipy.spatial import cKDTree
import networkx as nx
import time

# ─────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────
API_KEY        = "30C56F5E-1932-356E-9A50-D2979BEECA16"
REGION_LIST    = ["강원도 춘천시", "강원도 화천군"]
START_ADDRESS = "강원특별자치도 춘천시 공지로 255"  # 시작 주소
ITEM_COLS      = ["사과", "감자"]
ITEM_TO_GROUP  = {0: 0, 1: 0}
GROUP_CAP      = {0: 400, 1: 200}

# ─────────────────────────────────────
# 0. 입력 데이터
# ─────────────────────────────────────
input_data = [
    {"address":"강원특별자치도 춘천시 백령로 156","type":"customer","사과":100,"감자":50},
    {"address":"강원특별자치도 춘천시 강원대학길 1","type":"warehouse","사과":100,"감자":50},
    {"address":"강원특별자치도 춘천시 공지로 255","type":"customer","사과":0,"감자":0},
]
df_input = pd.DataFrame(input_data)

# ─────────────────────────────────────
# 1. 법정동 코드 앞5자리
# ─────────────────────────────────────
def get_legal_code_prefix(area_name):
    resp = requests.get(
        "https://api.vworld.kr/req/address?",
        params=dict(service="address", request="getcoord", crs="epsg:4326",
                    address=area_name, format="json", type="road", key=API_KEY)
    )
    if resp.status_code == 200:
        try:
            return resp.json()['response']['refined']['structure']['level4AC'][:5]
        except KeyError:
            print("❌ 법정동 코드 없음:", area_name)
    else:
        print("❌ 요청 실패:", area_name, resp.status_code)
    return None

prefixes = [p for p in (get_legal_code_prefix(r) for r in REGION_LIST) if p]

# ─────────────────────────────────────
# 2. GPKG에서 링크/노드 불러오기
# ─────────────────────────────────────
gpkg = r"C:/Users/admin/Desktop/식품사막프로젝트/1지도프로젝트/shapefiles_package.gpkg"
link_q = " OR ".join(f"LEGLCD_SE LIKE '{p}%'" for p in prefixes)
node_q = " OR ".join(f"(LEGLCD_SE1 LIKE '{p}%') OR (LEGLCD_SE2 LIKE '{p}%')" for p in prefixes)

gdf_links = gpd.read_file(gpkg, layer="links_with_adj", where=link_q).to_crs("EPSG:5179")
gdf_nodes = gpd.read_file(gpkg, layer="nodes",         where=node_q).to_crs("EPSG:5179")

# ─────────────────────────────────────
# 3. 주소 → TM 좌표
# ─────────────────────────────────────
def geocode_tm(addr):
    resp = requests.get(
        "https://api.vworld.kr/req/address?",
        params=dict(service="address", request="getcoord", crs="epsg:5179",
                    address=addr, format="json", type="road", key=API_KEY)
    )
    if resp.status_code==200 and resp.json()['response']['status']=="OK":
        pt = resp.json()['response']['result']['point']
        return Point(float(pt['x']), float(pt['y']))
    print("❌ 주소 변환 실패:", addr)
    return None

df_input["point"]   = df_input["address"].apply(geocode_tm)
time.sleep(0.3)

# ─────────────────────────────────────
# 4. 최근접 노드 스냅 (KDTree)
# ─────────────────────────────────────
node_coords = np.vstack([geom.coords[0] for geom in gdf_nodes.geometry])
tree        = cKDTree(node_coords)
coords      = np.vstack([[pt.x, pt.y] for pt in df_input["point"]])
_, idxs     = tree.query(coords, k=1)

# 실제 node_id 리스트
snap_nodes  = list(gdf_nodes.iloc[idxs]["NF_ID"].values)
df_input["node_id"] = snap_nodes


pt = geocode_tm(START_ADDRESS)                                           # ① 주소 → 좌표
query_idx = tree.query([[pt.x, pt.y]], k=1)[1][0]                         # ② 전체 노드 기준 index
depot_node_id = gdf_nodes.iloc[query_idx]["NF_ID"]                       # ③ 해당 노드의 ID
DEPOT_INDEX = snap_nodes.index(depot_node_id)                            # ✅ 거리행렬에서 쓰는 인덱스




# ─────────────────────────────────────
# 5. 네트워크 그래프 구축
# ─────────────────────────────────────
G = nx.DiGraph()
for _, r in gdf_links.iterrows():
    u,v = r['BNODE_NFID'], r['ENODE_NFID']
    if pd.notnull(u) and pd.notnull(v):
        d = r.geometry.length
        if r.get('OSPS_SE','OWI002')=='OWI001':
            G.add_edge(u,v,weight=d)
        else:
            G.add_edge(u,v,weight=d)
            G.add_edge(v,u,weight=d)

# ─────────────────────────────────────
# 6. 거리행렬 계산
# ─────────────────────────────────────
dist_mat = pd.DataFrame(index=snap_nodes, columns=snap_nodes, dtype=float)
for u in snap_nodes:
    for v in snap_nodes:
        if u==v:
            dist_mat.loc[u,v]=0
        else:
            try:
                dist_mat.loc[u,v] = nx.shortest_path_length(G,u,v,'weight')
            except nx.NetworkXNoPath:
                dist_mat.loc[u,v] = np.inf

# ─────────────────────────────────────
# 7. net_demand 준비
# ─────────────────────────────────────
for i,item in enumerate(ITEM_COLS):
    df_input[f"net_{item}"] = df_input.apply(
        lambda r:  r[item] if r["type"]=="warehouse"
                  else -r[item] if r["type"]=="customer"
                  else 0,
        axis=1
    )

net_cols  = [f"net_{item}" for item in ITEM_COLS]
net_dem   = df_input.set_index("node_id").loc[snap_nodes, net_cols].to_numpy()

# ─────────────────────────────────────
# 8. Routing 함수 (node_id로 매핑)
# ─────────────────────────────────────
def greedy_balanced_route(dist_m, net_d, it2grp, grp_cap, depot=0, node2addr=None):
    N,K   = net_d.shape
    u      = net_d.copy().astype(float)
    total  = 0.0
    route  = []
    cur    = depot
    load_i = {k:0 for k in range(K)}
    load_g = {g:0 for g in grp_cap}
    while np.sum(np.abs(u))>0:
        cands=[]
        for i in range(N):
            for k in range(K):
                g=it2grp[k]; val=u[i,k]
                d=dist_m[cur,i]
                if val>0 and load_g[g]<grp_cap[g]:
                    q=min(val,grp_cap[g]-load_g[g]); score=q/(d or 1e-6)
                    cands.append((i,k,-q,d,score))
                elif val<0 and load_i[k]>0:
                    q=min(-val,load_i[k]); score=q/(d or 1e-6)
                    cands.append((i,k,q,d,score))
        if not cands: break
        i_s,k_s,q,d,_=max(cands,key=lambda x:x[4])
        g=it2grp[k_s]
        u[i_s,k_s]+=q
        if q<0: load_i[k_s]+= -q; load_g[g]+= -q
        else:   load_i[k_s]-= q; load_g[g]-= q
        total+=d
        # 실제 node_id로 저장
        route.append((snap_nodes[cur], snap_nodes[i_s], k_s, int(q), d))
        cur=i_s
    if cur!=depot:
        d=dist_m[cur,depot]; total+=d
        route.append((snap_nodes[cur], snap_nodes[depot], None, 0, d))

    df = pd.DataFrame(route, columns=["from_id","to_id","item_idx","qty","dist"])
    node2addr = node2addr or {}
    df["from_addr"] = df["from_id"].map(node2addr)
    df["to_addr"]   = df["to_id"].map(node2addr)
    df["item"]      = df["item_idx"].apply(lambda k: ITEM_COLS[int(k)] if pd.notna(k) else "")
    return df[["from_addr","to_addr","item","qty","dist"]], total


def format_route_output_smart_merge(route_df, total_distance):
    print("\n[경로 요약]")

    merged_routes = []
    buffer = None

    for _, row in route_df.iterrows():
        from_addr = row["from_addr"]
        to_addr   = row["to_addr"]
        dist      = row["dist"]
        item      = row["item"]
        qty       = row["qty"]

        if pd.isna(item) or item == "":
            # 복귀는 항상 별도 처리
            merged_routes.append({
                "from": from_addr, "to": to_addr,
                "dist": dist / 1000,
                "actions": ["복귀"]
            })
            buffer = None
            continue

        desc = f"{item} {abs(int(qty))}개 {'배송' if qty > 0 else '픽업'}"

        if buffer and from_addr == to_addr and dist == 0:
            # 같은 장소에서 추가 작업 (통합)
            buffer["actions"].append(desc)
        else:
            # 새 경로 시작
            buffer = {
                "from": from_addr,
                "to": to_addr,
                "dist": dist / 1000,
                "actions": [desc]
            }
            merged_routes.append(buffer)

    # 출력
    for i, r in enumerate(merged_routes, 1):
        actions_str = ", ".join(r["actions"])
        print(f"{i}. {r['from']} → {r['to']} ({r['dist']:.2f}km): {actions_str}")

    print(f"\n총 이동 거리: {total_distance / 1000:.2f} km")


# ─────────────────────────────────────
# 9. 실행 및 출력
# ─────────────────────────────────────
node2addr = df_input.set_index("node_id")["address"].to_dict()
route_df, tot = greedy_balanced_route(
    dist_mat.values, net_dem, ITEM_TO_GROUP, GROUP_CAP,
    depot=DEPOT_INDEX, node2addr=node2addr
)



format_route_output_smart_merge(route_df, tot)



