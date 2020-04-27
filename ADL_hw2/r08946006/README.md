# ADL_HW2_README

## sh 的使用順序
理論上都有按照助教的指示，以下依舊附上執行要點
以下請在解壓縮下之目錄運行：/r08946006

1. 請務必先執行完 sh ./download.sh，將會載下1個檔案，為：
    + best_model
2. 接著執行預測 shell file，為：
    + run.sh
3. 調用方式皆為以下，並請皆輸入"絕對路徑" (參照助教討論區敘述)
```bash=
bash ./<*.sh> <input_file> <output_file>
```

## 各個file是甚麼

1. run.sh
    + 調用 BERT_final_pred.py 生成預測
2. data_check.py
    + 生成各個 plot
3. BERT_final_pred.py
    + 生成預測
4. report.pdf
    + 就是個報告檔案
5. README.md
    + 就是個readme檔案