# EXP008

## 変更点
- 学習は追加せず、EXP003 と EXP007 の提出を重み付きラベル投票でブレンド
- 入力:
  - `EXP/EXP003/submission.csv` (0.50)
  - `EXP/EXP007/submission-007.csv` (0.50)
- 出力: `submission-008.csv`

## 実行
- `python EXP/EXP008/blend_submissions.py`
