import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTableWidget, QTableWidgetItem, QComboBox,
    QInputDialog, QMessageBox, QListWidget, QPushButton, QLineEdit, QStackedWidget
)
import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point
from scipy.spatial import cKDTree
import networkx as nx
import numpy as np
import time
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWidgets import QInputDialog
from PyQt5.QtWidgets import QFileDialog


# API 키 설정
API_KEY = "30C56F5E-1932-356E-9A50-D2979BEECA16"

import pulp

def run_partitioning(nodes, items, demand, vehicles, capacity, start_node, dist, item_to_group, groups):
    """
    구역분할(최적화) 실행 함수. 입력값은 모두 dict/list 등으로 전달.
    반환값: result_df (차량, 노드, 품목, 물량 등 DataFrame)
    """
    prob = pulp.LpProblem("Item_Based_Balanced_Cluster", pulp.LpMinimize)
    ratio = pulp.LpVariable.dicts("r", (nodes, items, vehicles), lowBound=0, upBound=1)

    # 각 차량별 총 처리량
    total_capacity = {k: sum(capacity[k].values()) for k in vehicles}
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

    # (1) 수요/공급이 존재하는 노드-품목에만 분배 제약 적용
    for i in nodes:
        for itm in items:
            if abs(demand[i][itm]) > 1e-4:
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
                            prob += ratio[i][itm][k] == 0

    # 목적함수: 편차 + 거리
    total_dist = pulp.lpSum([
        ratio[i][itm][k] * abs(demand[i][itm]) * dist.get((start_node[k], i), 1e6)
        for i in nodes for itm in items for k in vehicles
    ])
    alpha, beta = 1, 1
    prob += alpha * pulp.lpSum([deviation[k] for k in vehicles]) + beta * total_dist

    prob.solve()

    # 결과 정리
    rows = []
    for i in nodes:
        for itm in items:
            for k in vehicles:
                val = ratio[i][itm][k].varValue
                if val is not None and abs(val) > 1e-6:
                    q = val * demand[i][itm]
                    rows.append({
                        '노드': i,
                        '품목': itm,
                        '차량': k,
                        '물량': q,
                        '비율': val,
                        '온도군': item_to_group[itm],
                        '역할': '공급' if q > 0 else '수요' if q < 0 else '중립'
                    })
    result = pd.DataFrame(rows)
    if not result.empty:
        result['물량_int'] = result['물량'].round()
    else:
        result['물량_int'] = []
    return result


class RegionPage(QWidget):
    def __init__(self, on_confirm):
        super().__init__()
        self.on_confirm = on_confirm
        self.regions = []  # list of (sido_edit, sigungu_edit)
        layout = QVBoxLayout(self)

        btn_layout = QHBoxLayout()
        btn_add = QPushButton("＋ 지역 추가")
        btn_confirm = QPushButton("확인")
        btn_add.clicked.connect(self.add_row)
        btn_confirm.clicked.connect(self.confirm)
        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_confirm)
        btn_del = QPushButton("－ 지역 삭제")
        btn_del.clicked.connect(self.del_row)
        btn_layout.addWidget(btn_del)


        self.container = QVBoxLayout()
        layout.addLayout(self.container)
        layout.addLayout(btn_layout)
        
        self.add_row()  # 초기 한 행

    def add_row(self):
        h = QHBoxLayout()
        sido = QLineEdit(); sido.setPlaceholderText("시/도")
        sigungu = QLineEdit(); sigungu.setPlaceholderText("시군/구")
        h.addWidget(QLabel("시/도:")); h.addWidget(sido)
        h.addWidget(QLabel("시군/구:")); h.addWidget(sigungu)
        self.container.addLayout(h)
        self.regions.append((sido, sigungu))

    def del_row(self):
        if not self.regions:
            return
        layout_to_remove = self.container.takeAt(len(self.regions)-1)
        while layout_to_remove.count():
            item = layout_to_remove.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()
        self.regions.pop()




    def confirm(self):
        region_list = []
        for sido, sigungu in self.regions:
            s = sido.text().strip(); g = sigungu.text().strip()
            if s and g:
                region_list.append(f"{s} {g}")
        if not region_list:
            QMessageBox.warning(self, "경고", "최소 하나의 지역을 입력하세요.")
            return
        self.on_confirm(region_list)






class InputPage(QWidget):
    def __init__(self,region_list):
        super().__init__()
        self.region_list = region_list  # ← 멤버 변수로 저장
        self.address_list = []       # ← 여기에 추가

        layout = QVBoxLayout(self)

        # 테이블: 주소/유형 + 동적 품목
        self.table = QTableWidget(0, 1)  # 주소만
        self.table.setHorizontalHeaderLabels(["주소"])

        # Preview 및 차량 리스트
        main_hl = QHBoxLayout()
        main_hl.addWidget(self.table)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("▶ 등록된 품목 속성"))
        self.preview_items = QListWidget()
        right_panel.addWidget(self.preview_items)

        right_panel.addWidget(QLabel("▶ 등록된 차량"))
        # 차량 선택 콤보박스
        self.vehicle_combo = QComboBox()
        right_panel.addWidget(self.vehicle_combo)
        self.vehicle_combo.currentIndexChanged.connect(self.on_vehicle_changed)

        # ▶ 출발지 선택 콤보박스 추가
        right_panel.addWidget(QLabel("▶ 출발지 선택"))
        self.start_combo = QComboBox()
        right_panel.addWidget(self.start_combo)
        self.start_combo.currentIndexChanged.connect(self.on_start_changed)


        # 차량 추가/삭제 버튼
        btn_add_vehicle = QPushButton("차량 추가")
        btn_add_vehicle.clicked.connect(self.add_vehicle)
        right_panel.addWidget(btn_add_vehicle)


        btn_del_vehicle = QPushButton("차량 삭제")
        btn_del_vehicle.clicked.connect(self.del_vehicle)
        right_panel.addWidget(btn_del_vehicle)
        
        main_hl.addLayout(right_panel)
        layout.addLayout(main_hl)

        # 품목 속성 저장
        self.item_attrs = {}  # {item_name: {"temperature": ...}}
        # 차량 정보 저장
        self.vehicles = {}    # {vehicle_name: {"ambient":..,"refrigerated":..,"frozen":..}}

        # 버튼: 행/열/실행
        hl = QHBoxLayout()
        btn_add_row = QPushButton("행 추가")
        btn_add_row.clicked.connect(self.add_row)
        hl.addWidget(btn_add_row)

        btn_del_row = QPushButton("행 삭제")  # 🔹 행 삭제 버튼
        btn_del_row.clicked.connect(self.del_row)
        hl.addWidget(btn_del_row)

        btn_add_col = QPushButton("품목 추가")
        btn_add_col.clicked.connect(self.add_col)
        hl.addWidget(btn_add_col)


        btn_del_item = QPushButton("품목 삭제")  # 🔹 품목 삭제 버튼
        btn_del_item.clicked.connect(self.del_item)
        hl.addWidget(btn_del_item)

        btn_run = QPushButton("실행")
        btn_run.clicked.connect(self.run) 
        hl.addWidget(btn_run)

        # 샘플 데이터 입력 버튼 추가
        btn_sample = QPushButton("샘플 데이터 입력")
        btn_sample.clicked.connect(self.fill_sample_data)
        hl.addWidget(btn_sample)

        layout.addLayout(hl)

    def fill_sample_data(self):
        # 요청하신 노드/주소/수박/복숭아 데이터 자동 입력
        addresses = [
            "강원특별자치도 춘천시 마장길 39",
            "강원특별자치도 춘천시 사북면 용화산로 76-25",
            "강원특별자치도 춘천시 역전옛길 16 1층",
            "강원특별자치도 춘천시 후석로 248 춘성할인마트",
            "강원특별자치도 춘천시 남춘천새길 42 상가동 101호",
            "강원특별자치도 춘천시 춘천로 300",
            "강원특별자치도 춘천시 백령로 150",
            "강원특별자치도 춘천시 후석로228번길 47 상가",
            "강원특별자치도 춘천시 춘천로 326",
        ]
        items = ["수박", "복숭아"]
        # 노드별 수요/공급량
        data = [
            [40,    0],    # 0
            [  0,  30],    # 1
            [ -5,   -3],    # 2
            [ -5,   -5],    # 3
            [ -5,   -4],    # 4
            [-10,   -5],    # 5
            [-10,   -5],    # 6
            [ -2,   -3],    # 7
            [ -3,   -5],    # 8
        ]
        self.table.setRowCount(len(addresses))
        self.table.setColumnCount(1 + len(items))
        self.table.setHorizontalHeaderLabels(["주소"] + items)
        for r, addr in enumerate(addresses):
            self.table.setItem(r, 0, QTableWidgetItem(addr))
            for c, item in enumerate(items):
                self.table.setItem(r, c+1, QTableWidgetItem(str(data[r][c])))
        # 품목 속성(임의)
        self.item_attrs = {"수박": {"temperature": "실온"}, "복숭아": {"temperature": "실온"}}
        self.refresh_items()
        # 차량 샘플
        self.vehicles = {
            "차량1": {"ambient": 50, "refrigerated": 0, "frozen": 0, "start_addr": addresses[0]},
            "차량2": {"ambient": 20, "refrigerated": 0, "frozen": 0, "start_addr": addresses[1]},
        }
        self.refresh_vehicles()
        self.start_combo.clear()
        self.start_combo.addItems(addresses)


    def refresh_items(self):
        self.preview_items.clear()
        for item, attrs in self.item_attrs.items():
            self.preview_items.addItem(f"{item}: {attrs['temperature']}")

    def refresh_vehicles(self):
        """등록된 차량 목록을 vehicle_combo 콤보박스에 갱신합니다."""
        self.vehicle_combo.clear()
        for name, caps in self.vehicles.items():
            display = f"{name} (출발: {caps['start_addr']})"
            self.vehicle_combo.addItem(display, userData=name)


    def add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(""))
        self.update_address_list()

            
    def on_start_changed(self, idx):
        veh = self.vehicle_combo.currentData()
        if not veh:
            return
        addr = self.start_combo.currentText()
        # 선택된 차량의 start_addr을 갱신
        self.vehicles[veh]["start_addr"] = addr

    def del_row(self):
        row_count = self.table.rowCount()
        if row_count > 0:
            self.table.removeRow(row_count - 1)
        self.update_address_list()  # ← 추가
            
    





    def handle_depot_check(self, selected_row):
        for r in range(self.table.rowCount()):
            if r != selected_row:
                widget = self.table.cellWidget(r, 1)
                if isinstance(widget, QCheckBox):
                    widget.setChecked(False)


    def add_col(self):
        text, ok = QInputDialog.getText(self, "품목 추가", "새 품목명 입력:")
        if not (ok and text): return
        temp, ok2 = QInputDialog.getItem(
            self, f"{text} 속성 설정", "보관온도:", ["실온","냉장","냉동"], editable=False
        )
        if not ok2: return
        self.item_attrs[text] = {"temperature": temp}
        c = self.table.columnCount(); self.table.insertColumn(c)
        self.table.setHorizontalHeaderItem(c, QTableWidgetItem(text))
        for r in range(self.table.rowCount()):
            self.table.setItem(r, c, QTableWidgetItem(""))
        self.refresh_items()




    def del_item(self):
        if not self.item_attrs:
            QMessageBox.information(self, "안내", "삭제할 품목이 없습니다.")
            return

        item_name, ok = QInputDialog.getItem(
            self, "품목 삭제", "삭제할 품목 선택:", list(self.item_attrs.keys()), editable=False
        )
        if not ok or item_name not in self.item_attrs:
            return

        del self.item_attrs[item_name]

        for c in range(2, self.table.columnCount()):
            header = self.table.horizontalHeaderItem(c).text()
            if header == item_name:
                self.table.removeColumn(c)
                break


        self.refresh_items()



    def update_address_list(self):
        """테이블의 1열(주소)을 읽어 address_list 갱신"""
        self.address_list = []
        for r in range(self.table.rowCount()):
            item = self.table.item(r, 0)
            if item and item.text().strip():
                self.address_list.append(item.text().strip())



    def add_vehicle(self):
        # 1) 미리 주소 리스트 최신화
        self.update_address_list()

        # 2) 주소가 하나라도 있어야 차량 추가 가능
        if not self.address_list:
            QMessageBox.warning(self, "경고", "먼저 행 추가로 주소를 입력하세요.")
            return

        # 3) 차량 이름 입력
        name, ok = QInputDialog.getText(self, "차량 추가", "차량 이름 입력:")
        if not (ok and name):
            return

        # 4) 출발지 주소를 선택하도록 콤보박스 제공
        start_addr, ok_addr = QInputDialog.getItem(
            self, "출발지 선택", "차량의 출발지 주소를 선택하세요:",
            self.address_list, editable=False
        )
        if not ok_addr or not start_addr:
            return

        # 5) 용량 입력
        amb, ok1 = QInputDialog.getInt(self, "용량 설정", "실온 용량:")
        if not ok1: return
        ref, ok2 = QInputDialog.getInt(self, "용량 설정", "냉장 용량:")
        if not ok2: return
        fro, ok3 = QInputDialog.getInt(self, "용량 설정", "냉동 용량:")
        if not ok3: return

        # 6) vehicles에 저장
        self.vehicles[name] = {
            "ambient": amb,
            "refrigerated": ref,
            "frozen": fro,
            "start_addr": start_addr
        }

        # 7) 콤보박스에 추가
        self.vehicle_combo.addItem(name, userData=name)
        self.refresh_vehicles()
        # 출발지 콤보박스도 최신화
        self.start_combo.clear()
        self.start_combo.addItems(self.address_list)

    def del_vehicle(self):
        if not self.vehicles:
            QMessageBox.information(self, "안내", "삭제할 차량이 없습니다.")
            return

        name, ok = QInputDialog.getItem(
            self, "차량 삭제", "삭제할 차량을 선택하세요:",
            list(self.vehicles.keys()), editable=False
        )
        if ok and name:
            del self.vehicles[name]
            self.refresh_vehicles()


    def on_vehicle_changed(self, idx):
        # 차량이 바뀔 때마다 주소 리스트 갱신
        self.update_address_list()
        self.start_combo.clear()
        self.start_combo.addItems(self.address_list)
        # 기존에 저장된 출발지 선택
        veh = self.vehicle_combo.currentText()
        prev = self.vehicles.get(veh, {}).get("start_addr", "")
        if prev in self.address_list:
            self.start_combo.setCurrentText(prev)


    def run(self):
        sel_vehicle   = self.vehicle_combo.currentData()
        START_ADDRESS = self.start_combo.currentText()
        if not START_ADDRESS:
            QMessageBox.warning(self, "경고", "출발지 주소를 선택하세요.")
            return
        self.vehicles[sel_vehicle]["start_addr"] = START_ADDRESS
    
        # Step 1. 테이블 -> input_data
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        headers = [self.table.horizontalHeaderItem(c).text().strip() for c in range(cols)]
        input_data = []
        for r in range(rows):
            row = {}
            for c in range(cols):
                header = headers[c]
                if c == 0:
                    item = self.table.item(r, c)
                    txt = item.text().strip() if item else ""
                    row["address"] = txt
                elif c >= 1:
                    item = self.table.item(r, c)
                    txt = item.text().strip() if item else "0"
                    try:
                        row[header] = int(txt)
                    except ValueError:
                        row[header] = 0
            if row.get("address"):
                input_data.append(row)

        # Step 2. 품목명 및 속성 추출/검증
        non_item_keys = {"address", "type"}
        ITEM_COLS = sorted(set().union(*(row.keys() for row in input_data)) - non_item_keys)
        missing_items = [item for item in ITEM_COLS if item not in self.item_attrs]
        if missing_items:
            QMessageBox.warning(self, "입력 오류", f"다음 품목의 속성(온도)이 누락되었습니다: {', '.join(missing_items)}.")
            return

        # Step 3. 거리 행렬 우선 계산
        config = {
            "API_KEY": self.api_key,
            "GPKG_PATH": self.gpkg_path,
            "REGION_LIST": self.region_list,
            "START_ADDRESS": START_ADDRESS,
            "ITEM_COLS": ITEM_COLS,
            "input_data": input_data,
        }
        try:
            df_input, dist_mat, snap_nodes, depot_node_id, G, network_issues = prepare_network_data(config)
            config["input_data"] = df_input.to_dict('records')
        except Exception as e:
            QMessageBox.critical(self, "네트워크 오류", f"거리 행렬 계산 중 심각한 오류 발생: {e}")
            return

        # Step 4. 구역분할 최적화 실행
        node_list = list(df_input.index)
        address_to_node = {addr: idx for idx, addr in zip(node_list, df_input['address'])}
        nodes = node_list
        items = ITEM_COLS
        demand = {idx: {item: int(df_input.loc[idx, item]) if item in df_input.columns else 0 for item in items} for idx in nodes}
        vehicles = list(range(len(self.vehicles)))
        vehicle_names = list(self.vehicles.keys())
        group_names = ['실온','냉장','냉동']
        capacity = {
            v: {g: self.vehicles[vehicle_names[v]][k] for g, k in zip(group_names, ['ambient','refrigerated','frozen'])}
            for v in vehicles
        }
        start_node = {v: address_to_node[self.vehicles[vehicle_names[v]]['start_addr']] for v in vehicles}
        
        dist_mat_np = dist_mat.values
        dist = {(i, j): dist_mat_np[i, j] for i in nodes for j in nodes}

        item_to_group = {item: self.item_attrs.get(item, {}).get('temperature', '실온') for item in items}
        groups = list(set(item_to_group.values()))

        try:
            result_df = run_partitioning(nodes, items, demand, vehicles, capacity, start_node, dist, item_to_group, groups)
        except Exception as e:
            QMessageBox.warning(self, "구역분할 오류", str(e))
            return

        # Step 5. 경로계산 함수 실행
        config.update({
            "partition_df": result_df,
            "VEHICLES": self.vehicles,
            "GROUPS": groups,
            "ITEM_TO_GROUP": item_to_group,
        })

        route_df, tot, issues = calculate_routes_from_partition(
            config, df_input, dist_mat, snap_nodes, depot_node_id, G, network_issues
        )
        
        if issues:
            QMessageBox.warning(self, "경고: 입력/전처리 문제", "다음 문제가 감지되었습니다:\n" + "\n".join(issues[:10]))
        
        summary = format_all_routes(route_df, tot)
        QMessageBox.information(self, "경로 요약", summary)


class SettingsPage(QWidget):
    def __init__(self, on_confirm):
        super().__init__()
        self.on_confirm = on_confirm
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("🔑 VWorld API 키 입력:"))
        self.api_input = QLineEdit()
        self.api_input.setPlaceholderText("예: 30C56F5E-1932-356E-9A50-D2979BEECA16")
        layout.addWidget(self.api_input)

        layout.addWidget(QLabel("📁 결과 저장 경로 선택:"))
        hlayout = QHBoxLayout()
        self.path_input = QLineEdit()
        btn_browse = QPushButton("찾아보기")
        btn_browse.clicked.connect(self.browse_file)
        hlayout.addWidget(self.path_input)
        hlayout.addWidget(btn_browse)
        layout.addLayout(hlayout)

        btn_confirm = QPushButton("확인")
        btn_confirm.clicked.connect(self.confirm)
        layout.addWidget(btn_confirm)

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "GPKG 파일 선택",
            "",
            "GeoPackage 파일 (*.gpkg);;모든 파일 (*)"
        )
        if file_path:
            self.path_input.setText(file_path)


    def confirm(self):
        api_key = self.api_input.text().strip()
        gpkg_path = self.path_input.text().strip()

        if not api_key or not gpkg_path:
            QMessageBox.warning(self, "경고", "API 키와 GPKG 경로를 모두 입력하세요.")
            return

        self.on_confirm(api_key, gpkg_path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("물류 경로 계산기")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.api_key = None
        self.save_path = None
        self.region_list = []

        self.settings_page = SettingsPage(self.on_settings_confirm)
        self.stack.addWidget(self.settings_page)

    def on_settings_confirm(self, api_key, gpkg_path):
        self.api_key = api_key
        self.gpkg_path = gpkg_path

        self.region_page = RegionPage(self.on_region_confirm)
        self.stack.addWidget(self.region_page)
        self.stack.setCurrentWidget(self.region_page)

    def on_region_confirm(self, region_list):
        self.region_list = region_list
        self.input_page = InputPage(region_list)

        self.input_page.api_key = self.api_key
        self.input_page.gpkg_path = self.gpkg_path
        self.stack.addWidget(self.input_page)
        self.stack.setCurrentWidget(self.input_page)


def prepare_network_data(config):
    """
    주소 목록과 네트워크 파일(GPKG)을 기반으로 지오코딩, 노드 스냅,
    거리 행렬 계산 등 라우팅에 필요한 사전 준비 작업을 수행합니다.
    """
    API_KEY        = config["API_KEY"]
    GPKG_PATH      = config["GPKG_PATH"]
    REGION_LIST    = config["REGION_LIST"]
    START_ADDRESS  = config["START_ADDRESS"]
    ITEM_COLS      = config["ITEM_COLS"]
    input_data     = config["input_data"]
    df_input       = pd.DataFrame(input_data)
    issues         = []

    def get_legal_code_prefix(area_name):
        resp = requests.get(
            "https://api.vworld.kr/req/address?",
            params=dict(service="address", request="getcoord", crs="epsg:4326",
                        address=area_name, format="json", type="road", key=API_KEY)
        )
        if resp.status_code == 200:
            try:
                return resp.json()['response']['refined']['structure']['level4AC'][:5]
            except (KeyError, TypeError):
                issues.append(f"법정동 코드 없음: {area_name}")
        else:
            issues.append(f"법정동 코드 요청 실패 ({resp.status_code}): {area_name}")
        return None

    prefixes = [p for p in (get_legal_code_prefix(r) for r in REGION_LIST) if p]
    if not prefixes:
        raise RuntimeError("지역 리스트로부터 유효한 법정동 코드를 하나도 추출하지 못했습니다.")

    link_q = " OR ".join(f"LEGLCD_SE LIKE '{p}%'" for p in prefixes)
    node_q = " OR ".join(f"(LEGLCD_SE1 LIKE '{p}%') OR (LEGLCD_SE2 LIKE '{p}%')" for p in prefixes)

    try:
        gdf_links = gpd.read_file(GPKG_PATH, layer="links_with_adj", where=link_q).to_crs("EPSG:5179")
        gdf_nodes = gpd.read_file(GPKG_PATH, layer="nodes", where=node_q).to_crs("EPSG:5179")
    except Exception as e:
        raise RuntimeError(f"네트워크 레이어 로딩 실패: {e}")

    if gdf_nodes.empty or gdf_links.empty:
        raise RuntimeError("필터 조건에 해당하는 노드나 링크가 없습니다. 지역이나 GPKG 파일을 확인하세요.")

    def geocode_tm(addr):
        resp = requests.get(
            "https://api.vworld.kr/req/address?",
            params=dict(service="address", request="getcoord", crs="epsg:5179",
                        address=addr, format="json", type="road", key=API_KEY)
        )
        if resp.status_code==200 and resp.json().get('response', {}).get('status')=="OK":
            pt = resp.json()['response']['result']['point']
            return Point(float(pt['x']), float(pt['y']))
        return None

    points, failed_addrs = [], []
    for addr in df_input["address"]:
        pt = geocode_tm(addr)
        if pt:
            points.append(pt)
        else:
            failed_addrs.append(addr)
            points.append(None)
    df_input["point"] = points
    if failed_addrs:
        issues.append(f"다음 주소 지오코딩 실패: {', '.join(failed_addrs)}")
        df_input = df_input[df_input["point"].notna()].reset_index(drop=True)
        if df_input.empty:
            raise RuntimeError("모든 주소 지오코딩에 실패했습니다.")

    node_coords = np.vstack([geom.coords[0] for geom in gdf_nodes.geometry])
    tree = cKDTree(node_coords)
    coords = np.vstack([[pt.x, pt.y] for pt in df_input["point"]])
    _, idxs = tree.query(coords, k=1)
    snap_nodes = list(gdf_nodes.iloc[idxs]["NF_ID"].values)
    df_input["node_id"] = snap_nodes

    pt = geocode_tm(START_ADDRESS)
    if pt is None:
        depot_node_id = snap_nodes[0]
        issues.append(f"출발지 '{START_ADDRESS}' 지오코딩 실패. 첫 노드로 대체합니다.")
    else:
        query_idx = tree.query([[pt.x, pt.y]], k=1)[1][0]
        depot_node_id = gdf_nodes.iloc[query_idx]["NF_ID"]

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

    all_graph_nodes = set(snap_nodes)
    if depot_node_id not in all_graph_nodes:
        all_graph_nodes.add(depot_node_id)
    
    dist_mat = pd.DataFrame(index=snap_nodes, columns=snap_nodes, dtype=float)
    for u in snap_nodes:
        for v in snap_nodes:
            if u == v:
                dist_mat.loc[u, v] = 0
            else:
                try:
                    dist_mat.loc[u, v] = nx.shortest_path_length(G, u, v, 'weight')
                except nx.NetworkXNoPath:
                    dist_mat.loc[u, v] = np.inf
    
    inf_count = (dist_mat == np.inf).sum().sum()
    if inf_count > 0:
        issues.append(f"거리행렬에 경로 없음(INF) 쌍이 {int(inf_count)}개 있습니다.")

    return df_input, dist_mat, snap_nodes, depot_node_id, G, issues


def calculate_routes_from_partition(config, df_input, dist_mat, snap_nodes, depot_node_id, G, issues):
    """
    사전 계산된 거리 행렬과 구역 분할 결과를 바탕으로 차량별 상세 경로를 계산합니다.
    """
    ITEM_COLS = config["ITEM_COLS"]
    ITEM_TO_GROUP = config["ITEM_TO_GROUP"]
    START_ADDRESS = config["START_ADDRESS"]
    partition_df = config.get('partition_df')
    vehicles = config.get('VEHICLES', {})
    groups = config.get('GROUPS')

    for item in ITEM_COLS:
        df_input[f"net_{item}"] = df_input[item]
    net_cols  = [f"net_{item}" for item in ITEM_COLS]
    net_dem   = df_input.set_index("node_id").loc[snap_nodes, net_cols].to_numpy()

    def greedy_balanced_route(dist_m, net_d, it2grp, grp_cap, depot=0, node2addr=None):
        N,K = net_d.shape
        u = net_d.copy().astype(float)
        total = 0.0
        route = []
        cur = depot
        load_i = {k: 0 for k in range(K)}
        load_g = {g: 0 for g in grp_cap}

        while np.sum(np.abs(u)) > 1e-6:
            cands = []
            for i in range(N):
                for k in range(K):
                    val = u[i, k]
                    if abs(val) < 1e-6: continue
                    g = it2grp[k]
                    d = dist_m[cur, i]
                    
                    if val > 0 and load_g.get(g, 0) < grp_cap.get(g, 0):
                        q = min(val, grp_cap[g] - load_g[g])
                        score = q / (d or 1e-6)
                        cands.append((i, k, -q, d, score))
                    elif val < 0 and load_i[k] > 0:
                        q = min(-val, load_i[k])
                        score = q / (d or 1e-6)
                        cands.append((i, k, q, d, score))
            if not cands:
                break
            i_s,k_s,q,d,_=max(cands,key=lambda x:x[4])
            g=it2grp[k_s]
            u[i_s,k_s]+=q
            if q<0: load_i[k_s]+= -q; load_g[g]+= -q
            else:   load_i[k_s]-= q; load_g[g]-= q
            total+=d
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

    node2addr = df_input.set_index("node_id")["address"].to_dict()
    
    if partition_df is None or partition_df.empty or not vehicles:
        issues.append("구역분할 결과가 없거나 차량이 없어 단일 라우팅으로 대체합니다.")
        depot_idx = snap_nodes.index(depot_node_id)
        route_df, tot = greedy_balanced_route(
            dist_mat.values, net_dem, [ITEM_TO_GROUP[item] for item in ITEM_COLS],
            depot=depot_idx, node2addr=node2addr
        )
        return route_df, tot, issues

    if partition_df['차량'].dtype != object:
        vehicle_keys = list(vehicles.keys())
        partition_df['차량'] = partition_df['차량'].apply(lambda idx: vehicle_keys[int(idx)] if int(idx) < len(vehicle_keys) else idx)

    all_routes = []
    total_distance = 0
    for vname, vinfo in vehicles.items():
        vdf = partition_df[partition_df['차량'] == vname]
        if vdf.empty: continue

        v_net_dem = np.zeros_like(net_dem)
        for idx, node_id in enumerate(snap_nodes):
            for item_idx, item in enumerate(ITEM_COLS):
                row = vdf[(vdf['노드']==idx)&(vdf['품목']==item)]
                if not row.empty:
                    v_net_dem[idx, item_idx] = row['물량'].sum()
        
        v_start_addr = vinfo.get('start_addr', START_ADDRESS)
        try:
            v_depot_idx = df_input[df_input['address'] == v_start_addr].index[0]
        except IndexError:
            v_depot_idx = snap_nodes.index(depot_node_id)

        eng_to_kor = {'ambient': '실온', 'refrigerated': '냉장', 'frozen': '냉동'}
        raw_cap = vehicles[vname]
        grp_cap = {eng_to_kor[k]: v for k, v in raw_cap.items() if k in eng_to_kor}
        grp_cap = {g: grp_cap[g] for g in groups if g in grp_cap}
        
        it2grp = [ITEM_TO_GROUP[item] for item in ITEM_COLS]
        
        v_route_df, v_tot = greedy_balanced_route(
            dist_mat.values, v_net_dem, it2grp, grp_cap,
            depot=v_depot_idx, node2addr=node2addr
        )
        v_route_df['차량'] = vname
        all_routes.append(v_route_df)
        total_distance += v_tot

    if all_routes:
        route_df = pd.concat(all_routes, ignore_index=True)
        tot = total_distance
    else:
        route_df, tot = pd.DataFrame(), 0
        
    return route_df, tot, issues


def format_route_output_smart_merge(route_df, total_distance):
    msg = "[경로 요약]\n"
    merged_routes = []
    buffer = None

    for _, row in route_df.iterrows():
        from_addr = row["from_addr"]
        to_addr   = row["to_addr"]
        dist      = row["dist"]
        item      = row["item"]
        qty       = row["qty"]

        if pd.isna(item) or item == "":
            merged_routes.append({
                "from": from_addr, "to": to_addr,
                "dist": dist / 1000,
                "actions": ["복귀"]
            })
            buffer = None
            continue

        desc = f"{item} {abs(int(qty))}개 {'배송' if qty > 0 else '픽업'}"

        if buffer and from_addr == to_addr and dist == 0:
            buffer["actions"].append(desc)
        else:
            buffer = {
                "from": from_addr,
                "to": to_addr,
                "dist": dist / 1000,
                "actions": [desc]
            }
            merged_routes.append(buffer)

    for i, r in enumerate(merged_routes, 1):
        actions_str = ", ".join(r["actions"])
        msg += f"{i}. {r['from']} → {r['to']} ({r['dist']:.2f}km): {actions_str}\n"

    msg += f"\n총 이동 거리: {total_distance / 1000:.2f} km"
    return msg

def format_all_routes(route_df, total_distance):
    """
    차량이 여러 대면 각 차량별로 경로 요약을 출력합니다.
    """
    if '차량' not in route_df.columns or route_df['차량'].nunique() <= 1:
        return format_route_output_smart_merge(route_df, total_distance)
    
    msg = ""
    for vehicle, df in route_df.groupby('차량'):
        msg += f"\n🚚 {vehicle} 경로\n"
        msg += format_route_output_smart_merge(df, df['dist'].sum())
        msg += "\n"
    
    msg += f"\n\n[종합] 총 이동 거리: {total_distance / 1000:.2f} km"
    return msg.strip()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
