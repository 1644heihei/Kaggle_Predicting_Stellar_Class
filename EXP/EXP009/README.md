# EXP009

## 変更点
- 学習は行わず、**EXP007 の提出**（LB 0.96841）に redshift 後処理を適用
- ベースモデル: **007 のスタッキング出力**（`submission-007.csv`）
- 有効ルール（高純度のみ）:
  - `GALAXY` かつ `redshift > 1.5` → `QSO`（train 上 QSO 約 98.7%）
- 出力: `submission-009.csv`

## モデル選定
- **007 をそのまま使う**（003/008 ブレンドは LB で下振れ済み）
- 再学習は不要。007 の test 確率を保存してから flip する方式は今後の EXP010 以降向け

## 実行
```bash
python EXP/EXP009/redshift_postprocess.py
```

## オプションルール
`redshift_postprocess.py` の `OPTIONAL_RULES` を `RULES` に追加すると有効化できます。
- `QSO & z < 0.3 → GALAXY` は変更行数が多くリスク高め
