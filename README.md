# FairseqModelLoader

fairseqで作成されたモデルだけを，json形式で記述されたオプションから脳死で読み込んでコード内で使うためのライブラリです．  

# DEMO
詳細はsampleフォルダへ  


# Features
コマンドラインオプションではなくjson形式で扱い，純粋にmodelだけをfairseqから切り離して使いたいので作りました．  
modelの改変や拡張，特殊な処理をしたデータセット読み込ませたいなど，fairseq内で完結するにはキツい作業をするときににお使いください．  

# Requirement

詳細はrequrements.txtへ  

# Installation
まずfairseqのインストールを行う．https://github.com/pytorch/fairseq からリポジトリをダウンロードし，rootディレクトリで  
```bash
pip install --editable ./
```  
次に，このレポジトリをダウンロードし，rootディレクトリで  
```bash
pip install --editable ./
```

# Usage
```python
from FairseqModelLoader import loadModel
model = loadModel(your_config_file_path)
```  
configファイルの書き方はsample推奨

# Note
従来通りコマンドライン引数を渡しても動きます．（その場合はconfigファイルのパスを指定しないでください）  

# Author

* yaminchu
* NAIST
* tktkbohshi@gmail.com

# License
ライセンスを明示する

"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
