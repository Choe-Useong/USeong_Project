'''
유의사항
1. 이 코드가 포함되어 있는 파일을 직접 실행시켜야 함.
   (이 파일을 import해서 사용하면 실행 안됨.)
2. 모든 코드는 if __name__ == "__main__": 조건문 아래에서 작성해야 함.
'''
# 아래 조건문은 병렬환경 실행 시 오류를 방지하기 위함. 이 코드가 없으면 병렬환경 갯수만큼 순차 실행 (!= 병렬환경 학습)
if __name__ == "__main__": # 해당 코드가 구현된 파일이 직접 실행됐을 때만 실행
    
    import warnings
    # FutureWarning을 무시
    warnings.simplefilter(action='ignore', category=FutureWarning)

    import logging
    logging.getLogger('matplotlib.font_manager').disabled = True

    # -------------------------------
    # 1. 데이터 정의
    # -------------------------------
    import torch
    import numpy as np

    UNIT = 50

    fixed_net_demand = torch.tensor([
        [ 0,  0,  0],    # depot (0)
        [-300, -200,  200],    # node 1
        [-200,  400,  0],    # node 2
        [ 200, -300,  200],    # node 3
        [ 400, -200, -200],    # node 4
        [ 0,  100, -100],    # node 5
        [-100,  200, -100]     # node 6 (추가)
    ], dtype=torch.int32)

    fixed_net_demand = fixed_net_demand / UNIT  # 단위 변환
    
    dist_matrix = np.array([
        [0.0, 1.2, 2.5, 1.8, 2.0, 1.9, 2.2],
        [1.2, 0.0, 1.3, 1.5, 1.7, 2.1, 1.9],
        [2.5, 1.3, 0.0, 1.1, 0.9, 1.8, 2.3],
        [1.8, 1.5, 1.1, 0.0, 1.4, 2.0, 2.2],
        [2.0, 1.7, 0.9, 1.4, 0.0, 1.3, 1.5],
        [1.9, 2.1, 1.8, 2.0, 1.3, 0.0, 1.2],
        [2.2, 1.9, 2.3, 2.2, 1.5, 1.2, 0.0]
    ]) * 1000.0 
    item_to_group = [0, 0, 1]
    group_cap = {0: 400, 1: 200}
    group_cap = {g: cap // UNIT for g, cap in group_cap.items()}  # 단위 변환



    # -------------------------------
    # 2. 환경 클래스 정의
    # -------------------------------
    import gymnasium as gym
    from gymnasium.spaces import MultiDiscrete, Box
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from sklearn.manifold import MDS
    import matplotlib.pyplot as plt

    class VRPEnv(gym.Env):
        def __init__(self, fixed_net_demand, dist_matrix, item_to_group, group_cap):
            super().__init__()

            # 입력 데이터를 PyTorch 텐서로 변환
            self.fixed_net_demand = fixed_net_demand
            self.dist_matrix = torch.tensor(dist_matrix, dtype=torch.float32)
            self.item_to_group = torch.tensor(item_to_group, dtype=torch.int64)

            # 그룹 용량을 텐서로 변환
            self.num_groups = len(set(item_to_group))
            self.group_cap_tensor = torch.zeros(self.num_groups, dtype=torch.int32)

            for g, cap in group_cap.items():
                self.group_cap_tensor[g] = cap
            self.group_cap = group_cap
            self.num_nodes = fixed_net_demand.shape[0]
            self.num_items = fixed_net_demand.shape[1]
            self.max_steps = 300
            MAX_QTY = max(group_cap.values())  # 예: 400 -> UNIT=50 기준으로 8
            self.max_quantity_unit = MAX_QTY
            self.action_space = MultiDiscrete([
                self.num_nodes,
                self.num_items,
                2,                         # 작업 타입
                self.max_quantity_unit + 1  # 수량 단위 (0 ~ 8)
            ])

            # 그룹별 아이템 마스크 생성
            self.group_masks = []

            for g in range(self.num_groups):
                mask = (self.item_to_group == g)
                self.group_masks.append(mask)

            self.reset()
            obs_low = []
            obs_high = []

            for _ in range(self.num_nodes):
                # 수요 정규화: [-1.0, 1.0]
                obs_low.extend([-1.0] * self.num_items)
                obs_high.extend([1.0] * self.num_items)

                # 위치 one-hot: [0.0, 1.0]
                obs_low.append(0.0)
                obs_high.append(1.0)

                # 거리 정규화: [0.0, 1.0]
                obs_low.append(0.0)
                obs_high.append(1.0)

                # 도달 가능 여부: [0.0, 1.0]
                obs_low.append(0.0)
                obs_high.append(1.0)

            # 차량 적재량 (정규화됨): [0.0, 1.0]
            obs_low.extend([0.0] * self.num_items)
            obs_high.extend([1.0] * self.num_items)

            # 그룹 잔여 용량 (정규화됨): [0.0, 1.0]
            obs_low.extend([0.0] * self.num_groups)
            obs_high.extend([1.0] * self.num_groups)

            self.observation_space = Box(
                low=np.array(obs_low, dtype=np.float32),
                high=np.array(obs_high, dtype=np.float32),
                dtype=np.float32
            )


        def _get_total_unbalance(self):
            # 벡터화: 모든 노드(depot 제외)의 절대값 수요 합 계산
            return torch.sum(torch.abs(self.net_demand[1:])).item()
        
        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.net_demand = self.fixed_net_demand.clone()
            self.vehicle_capacity = torch.zeros(self.num_items, dtype=torch.int32)
            self.vehicle_pos = 0
            self.step_count = 0
            self.initial_unbalance = max(self._get_total_unbalance(), 1e-6)
            return self._get_obs(), {}
        
        def _get_obs(self):
        # 차량 위치 원-핫 인코딩
            vehicle_pos_onehot = torch.zeros(self.num_nodes, dtype=torch.float32)
            vehicle_pos_onehot[self.vehicle_pos] = 1.0

            # 모든 노드까지의 거리
            distances = self.dist_matrix[self.vehicle_pos].clone()
            max_dist = self.dist_matrix.max().item()

            # 그룹별 사용량 계산
            group_used = torch.zeros(self.num_groups, dtype=torch.int32)
            for g in range(self.num_groups):
                group_mask = self.group_masks[g]
                group_used[g] = torch.sum(self.vehicle_capacity[group_mask])

            # 그룹별 잔여 용량
            group_remaining = self.group_cap_tensor - group_used

            # 도달 가능 여부
            feasible = torch.zeros(self.num_nodes, dtype=torch.float32)
            feasible[0] = 1.0  # depot은 항상 가능

            for n in range(1, self.num_nodes):
                for i in range(self.num_items):
                    net_value = self.net_demand[n, i].item()
                    group_id = self.item_to_group[i].item()
                    can_pickup = (net_value > 0) and (group_remaining[group_id] >= 1)
                    can_deliver = (net_value < 0) and (self.vehicle_capacity[i] >= 1)
                    if can_pickup or can_deliver:
                        feasible[n] = 1.0
                        break

            # 관측 벡터 구성
            obs = []

            for i in range(self.num_nodes):
                # 1. 수요 정규화: 그룹 최대 용량으로
                raw_demand = self.net_demand[i].float()
                group_cap_vector = self.group_cap_tensor[self.item_to_group]
                norm_demand = (raw_demand / group_cap_vector).clamp(-1.0, 1.0)
                node_demand = norm_demand.tolist()

                # 2. 위치
                at_node = vehicle_pos_onehot[i].item()

                # 3. 거리 정규화
                norm_dist = distances[i].item() / max_dist

                # 4. 도달 가능 여부
                node_feasible = feasible[i].item()

                node_obs = node_demand + [at_node, norm_dist, node_feasible]
                obs.extend(node_obs)

            # 5. 차량 적재량 정규화
            norm_vehicle_capacity = (self.vehicle_capacity.float() / group_cap_vector).clamp(0.0, 1.0)
            obs.extend(norm_vehicle_capacity.tolist())

            # 6. 그룹 잔여 용량 정규화
            for g in sorted(self.group_cap.keys()):
                norm_remain = float(group_remaining[g].item()) / self.group_cap[g]
                obs.append(min(norm_remain, 1.0))

            return np.array(obs, dtype=np.float32)

        
        def step(self, action):
            prev_unbalance = self._get_total_unbalance()
            
            # 행동 해석: node, item, task_type, ratio_index
            node, item, task_type, amt_unit = map(int, action.tolist())
            amt = amt_unit

            # 거리 및 위치 업데이트
            dist = self.dist_matrix[self.vehicle_pos, node].item()
            self.vehicle_pos = node
            self.step_count += 1

            # 현재 상태값
            net = self.net_demand[node, item].item()
            cap = self.vehicle_capacity[item].item()
            group_id = self.item_to_group[item].item()
            group_mask = self.group_masks[group_id]
            group_used = torch.sum(self.vehicle_capacity[group_mask]).item()
            group_limit = self.group_cap[group_id]

            # 최대 가능 수량
            if task_type == 0:  # 픽업
                max_amt = min(net, group_limit - group_used)
            else:  # 배송
                max_amt = min(-net, cap)
            
            max_amt = max(0, max_amt)  # 음수 방지

            pickup = 0
            delivery = 0
            max_dist = self.dist_matrix.max().item()
            reward = -(dist / max_dist) * 10.0 

            if node != 0:
                if task_type == 0:  # 픽업
                    if net >= amt and (group_used + amt <= group_limit):
                        pickup = amt
                        self.net_demand[node, item] -= amt
                        self.vehicle_capacity[item] += amt
                    else:
                        reward -= 1.0  # 잘못된 픽업

                elif task_type == 1:  # 배송
                    if -net >= amt and cap >= amt:
                        delivery = amt
                        self.net_demand[node, item] += amt
                        self.vehicle_capacity[item] -= amt
                    else:
                        reward -= 1.0  # 잘못된 배송

            new_unbalance = self._get_total_unbalance()
            delta = prev_unbalance - new_unbalance
            reward += (delta / self.initial_unbalance) * 100.0
            reward -= 5
            terminated = bool(torch.all(self.net_demand[1:] == 0)) and self.vehicle_pos == 0
            truncated = self.step_count >= self.max_steps

            if terminated:
                reward += 10.0

            return self._get_obs(), reward, terminated, truncated, {
                "pickup": pickup,
                "delivery": delivery,
                "cap": self.vehicle_capacity.tolist(),
                "action": [node, item, task_type, amt],
                "dist": dist
            }


    # ---------------
    # 2. 병렬환경 구성
    # ---------------
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor

    # 환경을 생성하는 함수 선언
    def make_env():
        def _init():
            train_env = VRPEnv(fixed_net_demand, dist_matrix, item_to_group, group_cap)
            # Monitor을 이용하여 학습 때마다 ep_len_mean, ep_rew_mean을 출력
            return Monitor(train_env)
        return _init

    # range 내 숫자 = 사용할 cpu 코어 갯수
    # 사용하고자 하는 코어 갯수만큼 환경을 생성하여 병렬 학습
    train_env = SubprocVecEnv([make_env() for _ in range(8)])

    # -------------------------------
    # 3. 병렬환경에서 학습
    # -------------------------------
    from stable_baselines3.common.callbacks import EvalCallback

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs={"net_arch": [512] * 6},
        verbose=1,
        n_epochs=20,
        device = 'cpu',
        learning_rate=1e-4,
        batch_size=64,
    )
    
    eval_callback = EvalCallback(
    train_env,
    best_model_save_path="./ppo_vrp_model",  # 모델 저장 폴더
    log_path="./logs",                   # 로그 저장 경로
    eval_freq=10000,                     # 평가 주기 (스텝 수 기준)
    deterministic=True,
    render=False
    )

    model.learn(total_timesteps=2000000, 
            callback=eval_callback
            )
    
    # 병렬환경에서 학습된 모델을 저장
    model.save("ppo_vrp_model/vrp_info")
    
    # -------------------------------
    # 4. 평가 및 시각화
    # -------------------------------

    # 모델 평가는 단일환경에서 진행
    env = VRPEnv(fixed_net_demand, dist_matrix, item_to_group, group_cap)
    env = Monitor(env)

    obs, _ = env.reset()
    done = False
    route = []
    total_reward = 0.0
    total_dist = 0.0

    # 위에서 저장한 모델(병렬환경에서 학습된 모델)을 불러옴
    model = PPO.load("ppo_vrp_model/best_model.zip", device="cpu")

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        reward = reward
        total_reward += reward
        total_dist += info["dist"]
        route.append((info["action"], info["delivery"], info["pickup"], info["cap"], info["dist"], reward))

    print("\n상위 분류 기반 공유 용량 복합 행동 경로")

    for i, (a, d, p, cap, dist, reward) in enumerate(route):
        print(f"Step {i+1}: 노드 {a[0]}, 품목 {a[1]}, 작업 {'배송' if a[2]==1 else '픽업'}, 수량 {a[3]} → 배송 {d}, 픽업 {p}, 적재 {cap}, 이동거리 {dist:.2f}, 보상 {reward:.2f}")
    print(f"\n총 이동 거리: {total_dist:.2f}")
    print(f"누적 보상: {total_reward:.2f}")

    # 시각화
    coords = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(dist_matrix)
    node_locs = {i: tuple(coords[i]) for i in range(len(coords))}
    visited_nodes = [step[0][0] for step in route]

    plt.figure(figsize=(8, 6))
    plt.title("방문 경로 및 순서 방향 시각화")

    for i, (x, y) in node_locs.items():
        plt.scatter(x, y, c="red" if i == 0 else "blue", s=250, zorder=2)
        plt.text(x + 0.05, y + 0.05, f"Node {i}", fontsize=10, zorder=3)

    for i in range(len(visited_nodes) - 1):
        src = node_locs[visited_nodes[i]]
        dst = node_locs[visited_nodes[i + 1]]
        dx, dy = dst[0] - src[0], dst[1] - src[1]
        plt.arrow(src[0], src[1], dx * 0.9, dy * 0.9,
                  head_width=0.05, head_length=0.1, fc='green', ec='green', alpha=0.8, length_includes_head=True, zorder=1)
        mid_x = (src[0] + dst[0]) / 2
        mid_y = (src[1] + dst[1]) / 2
        plt.text(mid_x, mid_y, f"{i+1}", fontsize=9, color="darkgreen", zorder=4)

    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()




