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
        layout = QVBoxLayout(self)

        # 테이블: 주소/유형 + 동적 품목
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["주소", "유형", "출발지"])
        # Preview 및 차량 리스트
        main_hl = QHBoxLayout()
        main_hl.addWidget(self.table)

        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("▶ 등록된 품목 속성"))
        self.preview_items = QListWidget()
        right_panel.addWidget(self.preview_items)
        right_panel.addWidget(QLabel("▶ 등록된 차량"))
        self.preview_vehicles = QListWidget()
        right_panel.addWidget(self.preview_vehicles)

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


        layout.addLayout(hl)

    def refresh_items(self):
        self.preview_items.clear()
        for item, attrs in self.item_attrs.items():
            self.preview_items.addItem(f"{item}: {attrs['temperature']}")

    def refresh_vehicles(self):
        self.preview_vehicles.clear()
        for name, caps in self.vehicles.items():
            self.preview_vehicles.addItem(
                f"{name} - 실온:{caps['ambient']}, 냉장:{caps['refrigerated']}, 냉동:{caps['frozen']}"
            )

    def add_row(self):
        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(""))
        combo = QComboBox(); combo.addItems(["배송","적재"])
        self.table.setCellWidget(r, 1, combo)

            # 출발지 체크박스
        chk = QCheckBox()
        chk.stateChanged.connect(lambda _, row=r: self.handle_depot_check(row))
        self.table.setCellWidget(r, 2, chk)

        for c in range(2, self.table.columnCount()):
            self.table.setItem(r, c, QTableWidgetItem(""))

    def del_row(self):
        row_count = self.table.rowCount()
        if row_count > 0:
            self.table.removeRow(row_count - 1)







    def handle_depot_check(self, selected_row):
        for r in range(self.table.rowCount()):
            if r != selected_row:
                widget = self.table.cellWidget(r, 2)
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

        # 테이블에서 해당 열 제거
        for c in range(3, self.table.columnCount()):  # 0~2는 주소, 유형, 출발지
            header = self.table.horizontalHeaderItem(c).text()
            if header == item_name:
                self.table.removeColumn(c)
                break

        self.refresh_items()






    def add_vehicle(self):
        name, ok = QInputDialog.getText(self, "차량 추가", "차량 이름 입력:")
        if not (ok and name): return
        amb, ok1 = QInputDialog.getInt(self, "용량 설정", "실온 용량:")
        if not ok1: return
        ref, ok2 = QInputDialog.getInt(self, "용량 설정", "냉장 용량:")
        if not ok2: return
        fro, ok3 = QInputDialog.getInt(self, "용량 설정", "냉동 용량:")
        if not ok3: return
        self.vehicles[name] = {"ambient": amb, "refrigerated": ref, "frozen": fro}
        self.refresh_vehicles()


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








    def run(self):
        START_ADDRESS = None
        for r in range(self.table.rowCount()):
            widget = self.table.cellWidget(r, 2)
            if isinstance(widget, QCheckBox) and widget.isChecked():
                item = self.table.item(r, 0)
                START_ADDRESS = item.text().strip() if item else ""
                break

        if not START_ADDRESS:
            QMessageBox.warning(self, "경고", "출발지를 한 개 지정해야 합니다.")
            return
    
        # Step 1. 테이블 → input_data
        rows = self.table.rowCount()
        cols = self.table.columnCount()
        headers = [self.table.horizontalHeaderItem(c).text() for c in range(cols)]

        input_data = []
        for r in range(rows):
            row = {}
            for c in range(cols):
                header = headers[c]
                if c == 1:  # 유형
                    widget = self.table.cellWidget(r, c)
                    txt = widget.currentText() if widget else ""
                    row["type"] = "customer" if txt == "배송" else "warehouse"
                elif c == 0:  # 주소
                    item = self.table.item(r, c)
                    txt = item.text().strip() if item else ""
                    row["address"] = txt
                else:  # 품목 수량
                    item = self.table.item(r, c)
                    txt = item.text().strip() if item else "0"
                    try:
                        row[header] = int(txt)
                    except ValueError:
                        row[header] = 0
            if row.get("address"):
                input_data.append(row)

        df_input = pd.DataFrame(input_data)

        # Step 2. 품목명 추출
        non_item_keys = {"address", "type"}
        ITEM_COLS = sorted(set().union(*(row.keys() for row in input_data)) - non_item_keys)

        # Step 3. 온도 → 그룹번호 매핑
        temp_to_group = {"실온": 0, "냉장": 1, "냉동": 2}
        ITEM_TO_GROUP = {
            i: temp_to_group.get(self.item_attrs.get(item, {}).get("temperature", "실온"), 0)
            for i, item in enumerate(ITEM_COLS)
        }

        # Step 4. 차량 용량 → 그룹별 집계
        GROUP_CAP = {
            0: sum(v["ambient"] for v in self.vehicles.values()),
            1: sum(v["refrigerated"] for v in self.vehicles.values()),
            2: sum(v["frozen"] for v in self.vehicles.values())
        }

        # Step 5. 결과 출력
        msg = "[INPUT DATA]\n" + df_input.to_string(index=False)
        msg += "\n\n[ITEM_COLS]\n" + str(ITEM_COLS)
        msg += "\n\n[ITEM_TO_GROUP]\n" + str(ITEM_TO_GROUP)
        msg += "\n\n[GROUP_CAP]\n" + str(GROUP_CAP)

        QMessageBox.information(self, "구조 생성 결과", msg)
        
        # Step 6. config 딕셔너리 만들기
        config = {
            "API_KEY": self.api_key,               # ✅ 전달받은 값 사용
            "GPKG_PATH": self.gpkg_path,  # ✅ 추가
            "REGION_LIST": self.region_list,
            "START_ADDRESS": df_input.iloc[0]["address"],
            "ITEM_COLS": ITEM_COLS,
            "ITEM_TO_GROUP": ITEM_TO_GROUP,
            "GROUP_CAP": GROUP_CAP,
            "input_data": input_data
        }

        # Step 7. 경로계산 함수 실행
        route_df, tot = run_routing_pipeline(config)

        summary = format_route_output_smart_merge(route_df, tot)
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

        self.on_confirm(api_key, gpkg_path)  # ⬅️ 변수명 통일



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("물류 경로 계산기")
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.api_key = None
        self.save_path = None
        self.region_list = []

        # 🚀 앱 실행 시 SettingsPage가 첫 화면!
        self.settings_page = SettingsPage(self.on_settings_confirm)
        self.stack.addWidget(self.settings_page)

    def on_settings_confirm(self, api_key, gpkg_path):
        self.api_key = api_key
        self.gpkg_path = gpkg_path  # ✅ 저장

        self.region_page = RegionPage(self.on_region_confirm)
        self.stack.addWidget(self.region_page)
        self.stack.setCurrentWidget(self.region_page)


        # ⏩ Settings 완료되면 RegionPage로
        self.region_page = RegionPage(self.on_region_confirm)
        self.stack.addWidget(self.region_page)
        self.stack.setCurrentWidget(self.region_page)

    def on_region_confirm(self, region_list):
        self.region_list = region_list
        self.input_page = InputPage(region_list)

        self.input_page.api_key = self.api_key
        self.input_page.gpkg_path = self.gpkg_path  # ✅ 전달
        self.stack.addWidget(self.input_page)
        self.stack.setCurrentWidget(self.input_page)




def run_routing_pipeline(config):
    API_KEY        = config["API_KEY"]
    REGION_LIST    = config["REGION_LIST"]
    START_ADDRESS  = config["START_ADDRESS"]
    ITEM_COLS      = config["ITEM_COLS"]
    ITEM_TO_GROUP  = config["ITEM_TO_GROUP"]
    GROUP_CAP      = config["GROUP_CAP"]
    input_data     = config["input_data"]
    df_input       = pd.DataFrame(input_data)

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
    gpkg = config["GPKG_PATH"]
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

    # ─────────────────────────────────────
    # 9. 실행 및 출력
    # ─────────────────────────────────────
    node2addr = df_input.set_index("node_id")["address"].to_dict()
    route_df, tot = greedy_balanced_route(
        dist_mat.values, net_dem, ITEM_TO_GROUP, GROUP_CAP,
        depot=DEPOT_INDEX, node2addr=node2addr
    )

    return route_df, tot

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




if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
















