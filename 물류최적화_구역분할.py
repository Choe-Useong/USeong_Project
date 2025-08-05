import pulp
import pandas as pd
import numpy as np
from itertools import combinations

# ────────────────────────────────
# 1. 입력 데이터
# ────────────────────────────────
nodes = [0, 1, 2, 3, 4,5,6,7]


# 🔧 차량별 시작 노드 (노드 번호 기준)
start_node = {0: 0, 1: 2, 2: 4, 3: 6}  # 예: 차량 0은 노드 0, 차량 1은 노드 2, 차량 2는 노드 4에서 출발


# 1. 품목 및 그룹 정보
items = ['사과', '아이스크림', '감자', '요거트']
item_to_group = {
    '사과': '실온',
    '아이스크림': '냉동',
    '감자': '냉장',
    '요거트': '냉장'
}
groups = list(set(item_to_group.values()))
vehicles = [0, 1, 2,3]

# 2. 품목 단위 수요
demand = {
    0: {'사과': 0,  '아이스크림': 0,  '감자': 0,   '요거트': 0},     # 중립
    1: {'사과': -10, '아이스크림': 13, '감자': 0,   '요거트': 5},
    2: {'사과': 10,  '아이스크림': -5, '감자': 6,   '요거트': 5},
    3: {'사과': 0,   '아이스크림': -5, '감자': -5,  '요거트': -10},
    4: {'사과': 0,  '아이스크림': 0,  '감자': -1,   '요거트': 0},     # 중립
    5: {'사과': -5,  '아이스크림': -3,  '감자': 10,  '요거트': -5},
    6: {'사과': 5,   '아이스크림': 5,  '감자': -5,  '요거트': -5},
    7: {'사과': 0,   '아이스크림': -5, '감자': -5,  '요거트': 10}
}


# 3. 차량별 온도군 용량
capacity = {
    0: {'냉장': 5, '냉동': 5, '실온': 0},
    1: {'냉장': 5, '냉동': 0, '실온': 0},
    2: {'냉장': 0, '냉동': 3, '실온': 5},
    3: {'냉장': 0, '냉동': 0, '실온': 5}
}
total_capacity = {k: sum(capacity[k].values()) for k in vehicles}


# 거리 행렬 (정규화 포함)
# 🔄 거리행렬 대칭 생성
dist_raw = {}
for i in nodes:
    for j in nodes:
        if i == j:
            dist_raw[(i, j)] = 0
        elif (j, i) in dist_raw:
            dist_raw[(i, j)] = dist_raw[(j, i)]  # 대칭값 복사
        else:
            dist_raw[(i, j)] = np.random.randint(1, 100)

# 정규화
max_dist = max(dist_raw.values())
dist = {k: v / max_dist for k, v in dist_raw.items()}


# ────────────────────────────────
# 2. 모델 정의
# ────────────────────────────────
prob = pulp.LpProblem("Item_Based_Balanced_Cluster", pulp.LpMinimize)
ratio = pulp.LpVariable.dicts("r", (nodes, items, vehicles), lowBound=0, upBound=1)

# 각 차량별 총 처리량
vehicle_total = {
    k: pulp.lpSum([ratio[i][itm][k] * abs(demand[i][itm]) for i in nodes for itm in items])
    for k in vehicles
}
avg_usage_val = sum(total_capacity.values())
avg_usage = pulp.lpSum(vehicle_total[k] for k in vehicles) / avg_usage_val

# 편차 변수 정의
deviation = {k: pulp.LpVariable(f"dev_{k}", lowBound=0) for k in vehicles}
for k in vehicles:
    usage_k = vehicle_total[k] / total_capacity[k]
    prob += deviation[k] >= usage_k - avg_usage
    prob += deviation[k] >= avg_usage - usage_k

# ────────────────────────────────
# 3. 제약 조건
# ────────────────────────────────

# (1) 수요/공급이 존재하는 노드-품목에만 분배 제약 적용
for i in nodes:
    for itm in items:
        if abs(demand[i][itm]) > 1e-4:  # 수요나 공급이 있으면
            prob += pulp.lpSum([ratio[i][itm][k] for k in vehicles]) == 1


# (2) 각 차량-품목별 수급 균형
for k in vehicles:
    for itm in items:
        prob += pulp.lpSum([ratio[i][itm][k] * demand[i][itm] for i in nodes]) == 0

for k in vehicles:
    for g in groups:
        if capacity[k][g] == 0:
            for i in nodes:
                for itm in items:
                    if item_to_group[itm] == g:
                        prob += ratio[i][itm][k] == 0, f"No_{itm}_to_vehicle_{k}_with_zero_{g}_node_{i}"


# ────────────────────────────────
# 4. 목적함수 (편차 + 거리)
# ────────────────────────────────
# 목적함수: 차량별 시작노드로부터의 거리 × 물량
# dist에서 안전하게 가져오는 get() 사용
total_dist = pulp.lpSum([
    ratio[i][itm][k] * abs(demand[i][itm]) * dist.get((start_node[k], i), 1e6)
    for i in nodes for itm in items for k in vehicles
])


alpha, beta = 1, 1
prob += alpha * pulp.lpSum(deviation.values()) + beta * total_dist

# ────────────────────────────────
# 5. 최적화
# ────────────────────────────────
prob.solve()

# ────────────────────────────────
# 6. 결과 정리
# ────────────────────────────────
assign_df = []
for i in nodes:
    for itm in items:
        for k in vehicles:
            val = ratio[i][itm][k].varValue
            if val and val > 1e-4:
                amount = val * demand[i][itm]
                group = item_to_group[itm]
                assign_df.append([i, itm, k, group, round(val, 4), round(amount, 2)])


















result = pd.DataFrame(assign_df, columns=['노드', '품목', '차량', '온도군', '비율', '물량'])
result['역할'] = result['물량'].apply(lambda x: '공급' if x < 0 else ('수요' if x > 0 else '중립'))
# (1) 정수화 기본: round or floor
result['물량_int'] = result['물량'].round()

# (2) 노드-품목 단위로 원래 총 물량 대비 오차 계산
diff = (
    result.groupby(['노드', '품목'])['물량'].sum()
    - result.groupby(['노드', '품목'])['물량_int'].sum()
).round().astype(int)

# (3) 각 노드-품목마다 가장 물량 많은 차량에 delta 보정
for (node, item), delta in diff.items():
    if delta == 0:
        continue
    mask = (result['노드'] == node) & (result['품목'] == item)
    # 절댓값 물량 가장 큰 행
    target_idx = result.loc[mask, '물량'].abs().idxmax()
    result.loc[target_idx, '물량_int'] += delta

print(result)



check = (
    result.groupby(['노드', '품목'])[['물량', '물량_int']].sum().round()
)
check['차이'] = check['물량'] - check['물량_int']
print(check[check['차이'] != 0])














# 🚛 차량별 총 사용률
print("\n🚛 [2] 차량별 총 처리량 및 사용률")
for k in vehicles:
    assigned = result[result['차량'] == k]['물량'].abs().sum()
    usage = assigned / total_capacity[k]
    print(f"차량 {k}: 총 물량 = {assigned}, 용량 = {total_capacity[k]}, 사용률 = {usage:.2%}")

# 📦 차량-온도군별 처리량
print("\n📦 [3] 차량-온도군별 처리량 (절댓값 기준)")
pivot_vol = result.copy()
pivot_vol['abs_물량'] = pivot_vol['물량'].abs()
print(pivot_vol.pivot_table(index='차량', columns='온도군', values='abs_물량', aggfunc='sum', fill_value=0))

# 💡 차량-온도군별 사용률
print("\n💡 [4] 차량-온도군별 사용률 (%)")
for k in vehicles:
    print(f"\n차량 {k}:")
    for g in groups:
        cap = capacity[k][g]
        used = pivot_vol.query(f"차량=={k} & 온도군=='{g}'")['abs_물량'].sum()
        usage = used / cap if cap > 0 else 0
        print(f"  온도군 {g} – 사용량 {used:.1f} / 용량 {cap} → 사용률: {usage:.2%}")

# 📎 차량별 수요/공급 분배
print("\n📎 [5] 차량별 수요/공급 분배")
print(result.pivot_table(index='차량', columns='역할', values='물량', aggfunc='sum', fill_value=0))

# 🚗 차량별 물량×거리 총합
print("\n📦 차량별 물량 × 거리 총합")
vehicle_dist = {k: 0 for k in vehicles}
for row in result.itertuples():
    i = row.노드
    k = row.차량
    q = abs(row.물량)
    avg_dist = np.mean([dist[i, j] for j in nodes if i != j])
    vehicle_dist[k] += q * avg_dist
for k in vehicles:
    print(f"차량 {k}: {vehicle_dist[k]:.2f}")

# 🚙 차량별 담당 노드 간 평균 거리
print("\n🚙 [6] 차량별 담당 노드 간 평균 거리 (쌍별 평균 기준)")
vehicle_nodes = {
    k: set(result[result['차량'] == k]['노드'].unique()) for k in vehicles
}
for k in vehicles:
    nodes_k = list(vehicle_nodes[k])
    if len(nodes_k) < 2:
        print(f"차량 {k} → 담당 노드 수가 1개 이하이므로 평균 거리 계산 불가")
        continue
    pairwise_dists = [dist[i, j] for i, j in combinations(nodes_k, 2)]
    avg_pairwise_dist = np.mean(pairwise_dists)
    print(f"차량 {k} → 담당 노드 간 평균 거리: {avg_pairwise_dist:.4f} (정규화 기준)")


# ────────────────────────────────
# 🔍 [7] 차량별 클러스터 거리행렬 + 수요공급 출력 (물량_int 기준)
# ────────────────────────────────
print("\n🔍 [요청된 출력] 차량별 클러스터 거리행렬 및 노드별 수요/공급 정보")
for k in sorted(result['차량'].unique()):
    print(f"\n🚛 차량 {k} 클러스터:")

    # ── 해당 차량의 노드들
    cluster_nodes = sorted(result[result['차량'] == k]['노드'].unique())
    print(f"노드 목록: {cluster_nodes}")

    # ── 거리행렬 출력
    dist_mat = pd.DataFrame(index=cluster_nodes, columns=cluster_nodes, dtype=float)
    for i in cluster_nodes:
        for j in cluster_nodes:
            dist_mat.loc[i, j] = dist.get((i, j), np.nan)
    print("\n🧭 클러스터 내 거리행렬:")
    print(dist_mat.round(3))

    # ── 노드별 품목별 수요/공급 정보 출력
    df_sub = result[(result['차량'] == k) & (result['노드'].isin(cluster_nodes))]
    pivot_table = df_sub.pivot_table(
        index='노드',
        columns='품목',
        values='물량_int',
        aggfunc='sum',
        fill_value=0
    )
    print("\n📦 노드별 수요/공급 (물량_int 기준):")
    print(pivot_table)
