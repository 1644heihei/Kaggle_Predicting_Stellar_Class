# EXP007

## 変更点
- EXP005をベースに学習モデルを20本へ拡張
- モデル群: LGBM(5) / XGB(4) / Cat(3) / ET(1) / RF(1) / HGB(2) / Logistic(2) / KNN(2)
- 20モデル向けに、重み探索は全探索ではなくOOFスコア正規化で高速化
- 出力ファイル名: submission-007.csv

## 追加出力（test 確率）
学習・スタッキング実行後に以下も保存される:
- `test_proba-007.csv` — id, proba_GALAXY, proba_QSO, proba_STAR（STAR-tuned 後）
- `test_proba-007.npy` — 同上の確率配列 (n_test, 3)
- `artifacts/pred_test.npz`, `artifacts/oof.npz` — 再スタッキング用中間結果

## 実行
ノートブック `stellar_classification.ipynb` を実行（フル学習は約3時間）。
