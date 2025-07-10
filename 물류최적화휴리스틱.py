import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from imitation.algorithms.bc import BC
from imitation.data.types import Transitions
from vrp_env import VRPEnv  # ← 당신의 VRP 환경


# 테스트 실행 예시
fixed_net_demand = np.array([
    [   0,    0,    0],
    [-300, -200,  200], #1
    [-200,  400,    0], #2
    [ 200, -300,  200], #3
    [ 400, -200, -200], #4
    [   0,  100, -100], #5
    [-100,  200, -100]  #6
])

fixed_net_demand = fixed_net_demand  # 단위 변환


dist_matrix = np.array([
    [0.0,1.2,2.5,1.8,2.0,1.9,2.2],
    [1.2,0.0,1.3,1.5,1.7,2.1,1.9],
    [2.5,1.3,0.0,1.1,0.9,1.8,2.3],
    [1.8,1.5,1.1,0.0,1.4,2.0,2.2],
    [2.0,1.7,0.9,1.4,0.0,1.3,1.5],
    [1.9,2.1,1.8,2.0,1.3,0.0,1.2],
    [2.2,1.9,2.3,2.2,1.5,1.2,0.0]
])
item_to_group = [0, 0, 1]
group_cap = {0: 400, 1: 200}

def greedy_balanced_route(dist_matrix, net_demand, item_to_group, group_cap, depot=0):
    N, K = net_demand.shape
    u = net_demand.astype(float).copy()
    initial_unbal = np.sum(np.abs(u))
    total_dist = 0.0
    route = []
    current = depot

    load_by_item = {k: 0.0 for k in range(K)}
    load_by_group = {g: 0.0 for g in group_cap}

    while np.sum(np.abs(u)) > 0:
        candidates = []
        for i in range(N):
            if i == depot:
                continue
            for k in range(K):
                g = item_to_group[k]
                val = u[i, k]

                # 픽업: 공급 노드
                if val > 0 and load_by_group[g] < group_cap[g]:
                    q = min(val, group_cap[g] - load_by_group[g])
                    if q > 0:
                        score = q / dist_matrix[current, i] if dist_matrix[current, i] > 0 else q * 1e6
                        candidates.append((i, k, -q, q, dist_matrix[current, i], score))  # 음수는 픽업

                # 배송: 수요 노드 (★재고 확인 필수)
                elif val < 0 and load_by_item[k] > 0:
                    q = min(-val, load_by_item[k])
                    if q > 0:
                        score = q / dist_matrix[current, i] if dist_matrix[current, i] > 0 else q * 1e6
                        candidates.append((i, k, q, q, dist_matrix[current, i], score))  # 양수는 배송

        if not candidates:
            break

        best = max(candidates, key=lambda x: x[-1])
        i_sel, k_sel, signed_q, delta_u, dist, score = best
        g = item_to_group[k_sel]

        # 상태 갱신
        u[i_sel, k_sel] += signed_q

        if signed_q < 0:
            # 픽업
            load_by_item[k_sel] += -signed_q
            load_by_group[g] += -signed_q
        else:
            # 배송 (★ 재고 부족한 경우 무시되도록 보장됨)
            load_by_item[k_sel] -= signed_q
            load_by_group[g] -= signed_q

        total_dist += dist
        current = i_sel
        route.append((i_sel, k_sel, int(signed_q), delta_u, dist, score))

    if current != depot:
        return_dist = dist_matrix[current, depot]
        total_dist += return_dist
        # ⬇ 복귀 Step 기록 추가
        route.append((depot, 0, 0, 0.0, return_dist, 0.0))  # item=-1로 표시 (의미 없음), 수량=0


    final_unbal = np.sum(np.abs(u))
    total_reduction = initial_unbal - final_unbal
    df_route = pd.DataFrame(route, columns=["Node", "Item", "SignedQty", "ΔU", "Distance", "Score"])
    return df_route, total_dist, total_reduction



df_route, total_dist, total_reduction = greedy_balanced_route(
    dist_matrix, fixed_net_demand, item_to_group, group_cap
)

print(df_route.to_string(index=False))
print("총 이동 거리:", total_dist)
print("총 불균형 감소량:", total_reduction)




# 누적 적재량 추적
load_by_item = [0] * fixed_net_demand.shape[1]
cumulative_dist = 0.0

for step, row in df_route.iterrows():
    node = int(row["Node"])
    item = int(row["Item"])
    qty = int(row["SignedQty"])
    delta = row["ΔU"]
    dist = row["Distance"]
    score = row["Score"]
    cumulative_dist += dist

    task = "픽업" if qty < 0 else "배송"
    abs_qty = abs(qty)

    # 적재량 갱신
    if qty < 0:
        load_by_item[item] += abs_qty
    else:
        load_by_item[item] -= abs_qty

    # 작업 결과 출력
    print(f"Step {step + 1}: 노드 {node}, 품목 {item}, 작업 {task}, 수량 {abs_qty} → "
          f"배송 {0 if qty < 0 else abs_qty}, 픽업 {abs_qty if qty < 0 else 0}, "
          f"적재 {load_by_item}, 이동거리 {dist:.2f}, 누적거리 {cumulative_dist:.2f}")
