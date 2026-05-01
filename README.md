# 予防保全のための変圧器オイル温度予測 (Transformer Oil Temperature Prediction for Preventive Maintenance)

## 📌 プロジェクト概要
本プロジェクトは、電力変圧器の異常停止（ダウンタイム）を未然に防ぐ「予防保全」システムを想定した機械学習パイプラインの実装です。
過去の電力負荷データ（HUFL, HULL等）から将来のオイル温度（Oil Temperature: OT）を予測し、危険温度への到達を事前に検知して現場にアラートを出すことを目的としています。

単なる「精度の追求」にとどまらず、**「実運用におけるエラー（見逃しと誤報）のトレードオフ」**に踏み込み、現場のKPIを満たすビジネス志向のAIモデル（安全志向AI）を構築した点が特徴です。

## ✨ 主要な成果とビジネス上の結論

1. **実運用の壁の発見（1時間 vs 12時間）**
   ベースラインとして1時間先の予測を行いましたが、精度は高いものの「現場の対応猶予が短すぎる」という課題に直面。対応猶予を12時間に広げると、AIの優位性は証明されたものの「最大13℃の予測ブレ」が発生し、実用不可であることが判明しました。
   
2. **安全志向AIの開発（非対称損失関数の実装）**
   LightGBMの損失関数を独自に拡張し、「危険な温度上昇を見逃した場合、通常の5倍のペナルティを与える」アルゴリズムを実装。これにより、致命的な見逃し率を約3割削減し、AIの挙動をビジネス要件に合わせてコントロールできることを実証しました。

3. **「3時間先予測」という最適解の発見**
   「現場の猶予時間」と「AIの精度」のスイートスポットを探索した結果、**3時間先予測モデル**に行き着きました。
   AIの予測値に対して **5.4℃の安全マージン** を設定することで、**「異常の見逃しゼロ（100%安全）」と「誤報率10%未満」を両立**させることに成功し、実稼働可能なシステム要件を定義しました。

## 📁 ディレクトリ構造とパイプライン

```bash
.
├── 01_eda.py                          # 探索的データ分析（波形トレンドと相関の可視化）
├── 02_preprocess.py                   # スライディングウィンドウ生成・スケーリング
├── 03_modeling.py                     # ベースモデルの学習（Optunaによるハイパーパラメータチューニング）
├── 04_evaluation.py                   # 全モデル・データセットの評価指標集計
├── 05_predict_12h_experiment.py       # 12時間先予測の検証（AIの限界検証）
├── 06_asymmetric_loss_experiment.py   # 非対称損失関数（安全志向AI）の実装
├── 08_predict_6h_experiment.py        # 最適解探索（6時間/3時間先予測のシミュレーション基盤）
├── requirements.txt                   # 依存ライブラリ
└── presentation_images/               # プレゼン用グラフ出力先
```

## 🚀 実行方法 (How to Run)

### 1. 環境構築
Python 3.9以上を推奨します。依存ライブラリをインストールしてください。
```bash
pip install -r requirements.txt
```

### 2. データセットの配置
本リポジトリには容量の都合上、元のCSVデータは含まれていません。
プロジェクトのルートに `data` フォルダを作成し、以下の4つのファイルを配置してください。
* `ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`

### 3. パイプラインの実行
各スクリプトを番号順に実行することで、データの前処理からモデルの評価までが自動で行われます。
```bash
python 01_eda.py
python 02_preprocess.py
python 03_modeling.py
python 04_evaluation.py
```
結果やグラフは `evaluation_results` ディレクトリに出力されます。

## 📊 使用技術
* **Language**: Python
* **Data Processing**: pandas, NumPy, scikit-learn
* **Machine Learning**: LightGBM, Ridge Regression
* **Hyperparameter Tuning**: Optuna
* **Visualization**: Matplotlib, Seaborn
