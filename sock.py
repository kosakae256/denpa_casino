import socket
from time import sleep
import cv2
import glob
from PIL import Image
import numpy as np
import os
import json
import concurrent.futures

IP_ADDRESS = '192.168.0.104'

character_list = glob.glob("./chara/*.png")
with open('data.json', 'r', encoding='utf-8') as file:
    match = json.load(file)

# キャプチャデバイスのID (通常は0、外部デバイスの場合は1など)
capture_device_id = 0

# キャプチャオブジェクトを作成
cap = cv2.VideoCapture(capture_device_id)

# カメラが正しく開けているか確認
if not cap.isOpened():
    print(f"キャプチャデバイスID {capture_device_id} を開けません。デバイスIDを確認してください。")
    exit()

# フォーマット、フレームサイズ、フレームレートの設定
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

def display_video():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # フレームを表示
        cv2.imshow('Video', frame)

        # 'q'キーが押されたら表示を終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def send_message(message):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((IP_ADDRESS, 65432))
    client_socket.sendall(message.encode())
    response = client_socket.recv(1024)
    client_socket.close()

def calculate_image_similarity(image1, image2):
    """
    2つの画像ファイルの類似度を計算する関数。

    Args:
        image1_path (str): 1つ目の画像ファイルのパス
        image2_path (str): 2つ目の画像ファイルのパス

    Returns:
        float: 画像の類似度（0から1の範囲）
    """
    # 画像サイズが同じであることを確認
    if image1.shape != image2.shape:
        raise ValueError("画像サイズが一致しません。")

    # ピクセルごとの差分を計算
    difference = cv2.absdiff(image1, image2)

    # 差分の合計を計算
    total_difference = np.sum(difference)

    # 最大可能差分（すべてのピクセルが最大差分（255）の場合）
    max_difference = image1.size * 255

    # 類似度の正規化
    similarity = 1 - (total_difference / max_difference)
    return similarity


class Control():
    def __init__(self):
        """
        flg0: A連打
        flg1: カジノで目の前に行く
        flg2: 
        """
        self.flg = "flg0"

    def detect_bridge(self):
        ret, frame = cap.read()
        bridge_cut = cv2.imread("./bridge_cut.png")
        mov_cut = frame[350:600, 280:350]
        # グレースケールに変換する
        gray1 = cv2.cvtColor(bridge_cut, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(mov_cut, cv2.COLOR_BGR2GRAY)

        # ヒストグラムを計算する
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0,256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0,256])

        # ヒストグラムの類似度を計算する
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        if similarity>0.99:
            return True

    def detect_gameend(self):
        ret, frame = cap.read()
        gameend_cut = cv2.imread("./gameend_cut.png")[:, 0:5]
        mov_cut = frame[25:45, 560:565]
        
        # BGRからHSVに変換
        hsv_image = cv2.cvtColor(mov_cut, cv2.COLOR_BGR2HSV)

        # 黄色の範囲を定義 (HSV空間)
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([40, 160, 255])

        # 黄色のピクセルを抽出
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # 黄色のピクセル数をカウント
        yellow_pixels = cv2.countNonZero(yellow_mask)

        # 画像全体のピクセル数を取得
        total_pixels = mov_cut.shape[0] * mov_cut.shape[1]

        # 黄色の割合を計算
        yellow_ratio = yellow_pixels / total_pixels

        # 黄色の割合が閾値を超えた場合はTrueを返す
        return not yellow_ratio > 0.6

    # A連打(橋を検出したらflg1に以降)
    def flg0(self):
        print("homeに戻って橋を検知するまでAボタン")
        send_message('HOME 0.1s\n0.1s')
        sleep(1)
        send_message('X 0.1s')
        c = 0
        while self.flg=="flg0":
            c+=1
            sleep(0.1)
            f = self.detect_bridge()
            if f:
                self.flg = "flg1"
                return
            send_message("A 0.1s")
            # 1分間見つからなかったらシステム的に異常なので、リセットをかける
            if c==600:
                self.flg = "flg0"
                return
    
    # 闘技場へアクセス
    def flg1(self):
        print("闘技場へ移動")
        sleep(1)
        send_message('L 0.1s')
        sleep(1)
        send_message('DPAD_LEFT 0.1s')
        sleep(0.3)
        send_message('DPAD_LEFT 0.1s')
        sleep(0.3)
        send_message('DPAD_DOWN 0.1s')
        sleep(0.3)
        send_message('A 0.1s')
        sleep(2.0)
        send_message('DPAD_LEFT 2.5s')
        send_message('DPAD_UP 1.5s')
        send_message('A 0.1s')
        sleep(2)
        send_message('A 0.1s')
        sleep(2)
        send_message('A 0.1s')
        sleep(0.5)
        self.flg = "flg2"
        return

    def flg2(self):
        match_number, l = self.templete_match()
        # 不明なマッチ(おそらく変な場所に行ってる)の場合はリセット(flg0)をかける
        if match_number==-1:
            print("不明なマッチ - リセット")
            self.flg = "flg0"
            for i in range(20):
                send_message("A 0.1s")
                sleep(0.1)
            return
        
        # もし最初に賭けるべき対象に選ばれていなかったらスキップする
        if not match["data"][match_number]["first"]:
        # if False:
            pass # first検知をスキップする用
            self.flg = "flg0"
            for i in range(20):
                send_message("A 0.1s")
                sleep(0.1)
            return
        # 賭けるべき対象なら
        else:
            target = match["data"][match_number]["bet"]
            print(target + "に賭ける")
            ind = l.index(target)

            # indの数だけ右に移動する
            for _ in range(ind):
                send_message('DPAD_RIGHT 0.1s')
                sleep(0.4)
            send_message('A 0.1s')
            sleep(0.5)
            send_message("DPAD_LEFT 0.1s")
            sleep(0.4)
            send_message("DPAD_LEFT 0.1s")
            sleep(0.4)
            send_message("A 0.1s")
            self.flg = "flg3"
            return
    
    # ゲーム状況を判断しながら操作する
    def flg3(self):
        for i in range(50):
            send_message("A 0.1s")
            sleep(0.1)
        while True:
            sleep(0.1)
            # ゲームが終了していたら
            if self.detect_gameend():
                print("ゲーム終了を検出")
                sleep(5)
                send_message("A 0.1s")
                sleep(3)
                send_message("A 0.1s")
                sleep(5)
                ret, frame = cap.read()
                img = cv2.imread("./kakutou_cut.png")
                mssim, ssim = cv2.quality.QualitySSIM_compute(img, frame[10:50, 148:208])
                # 負けてる場合
                if (mssim[0] + mssim[1] + mssim[2])/3 > 0.90:
                    print("敗北")
                    send_message("A 0.1s")
                    sleep(1)
                    self.flg="flg2"
                    return
                else:
                    print("勝利")
                    send_message("A 0.1s")
                    sleep(0.5)
                    send_message("A 0.1s")
                    sleep(7)
                    send_message("A 0.1s")
                    sleep(1)
                    match_number, l = self.templete_match()
                    if (match_number==-1):
                        print("不明なマッチ - 1番に賭ける")
                        return
                    target = match["data"][match_number]["bet"]
                    print(target + "に当たった分を賭ける")
                    ind = l.index(target)

                    # indの数だけ右に移動する
                    for _ in range(ind):
                        send_message('DPAD_RIGHT 0.1s')
                        sleep(0.4)
                    self.flg = "flg3"
                    return


    # どの試合かを検出する
    def templete_match(self):
        ret, frame = cap.read()
        sim_1 = ""
        sim_2 = ""
        sim_3 = ""
        sim_4 = ""
        sim_1_score = 0
        sim_2_score = 0
        sim_3_score = 0
        sim_4_score = 0

        for character in character_list:
            filename = os.path.basename(character).split(".")[0]
            img = Image.open(character)
            image = np.asarray(img)[:, :, ::-1]
            tmp_sim1 = calculate_image_similarity(image, frame[125:225, 53:123])
            tmp_sim2 = calculate_image_similarity(image, frame[125:225, 149:219])
            tmp_sim3 = calculate_image_similarity(image, frame[125:225, 243:313])
            tmp_sim4 = calculate_image_similarity(image, frame[125:225, 338:408])
            # tmp_sim1 = (mssim_1[0] + mssim_1[1] + mssim_1[2])/3
            # tmp_sim2 = (mssim_2[0] + mssim_2[1] + mssim_2[2])/3
            # tmp_sim3 = (mssim_3[0] + mssim_3[1] + mssim_3[2])/3
            # tmp_sim4 = (mssim_4[0] + mssim_4[1] + mssim_4[2])/3
            sim_1_score = max(sim_1_score, tmp_sim1)
            sim_2_score = max(sim_2_score, tmp_sim2)
            sim_3_score = max(sim_3_score, tmp_sim3)
            sim_4_score = max(sim_4_score, tmp_sim4)
            if sim_1_score == tmp_sim1: sim_1 = filename
            if sim_2_score == tmp_sim2: sim_2 = filename
            if sim_3_score == tmp_sim3: sim_3 = filename
            if sim_4_score == tmp_sim4: sim_4 = filename
        
        match_list = [sim_1, sim_2, sim_3, sim_4]
        match_list = [item for item in match_list if item != "ぐりーん"]
        print(match_list)

        tmp_match_2 = sorted(match_list)
        for i in range(len(match["data"])):
            tmp_match = sorted(match["data"][i]["monsters"])
            if tmp_match == tmp_match_2:
                return i, match_list
        return -1, []

    
    def exec(self):
        while True:
            try:
                if (self.flg=="flg0"):
                    self.flg0()
                elif (self.flg=="flg1"):
                    self.flg1()
                elif (self.flg=="flg2"):
                    self.flg2()
                elif (self.flg=="flg3"):
                    self.flg3()
            except:
                self.flg=="flg0"
                for i in range(50):
                    send_message("A 0.1s")
                    sleep(0.1)
                sleep(5)

if __name__ == "__main__":
    # ThreadPoolExecutorを使用して並列実行
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 動画表示を別スレッドで実行
        executor.submit(display_video)
        control = Control()
        control.exec()