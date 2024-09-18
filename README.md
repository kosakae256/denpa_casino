windows側ではgit、python3のinstallが必要です
python3は3.12で動作確認してますが、単に画像処理やってるだけなのでそれに近ければ動きます

### windows側

cdで作業ディレクトリに移動
```
git clone https://github.com/kosakae256/denpa_casino
cd denpa_casino
pip install -r requirements.txt
```

### ubuntu側

![image](https://github.com/user-attachments/assets/3fd7d807-8d1f-41e9-a5eb-0b4fd2634871)

```
ip addr
```
をubuntu側で実行して、inet 192.168.0.xxxの部分を覚える
sock.pyの11行目を覚えたものに置き換える

### windows側 実行
```
python sock.py
```

