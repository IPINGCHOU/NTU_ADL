# ADL_HW1_README

## sh 的使用順序
理論上都有按照助教的指示，以下依舊附上執行要點
以下請在解壓縮下之目錄運行：/r08946006

1. 請務必先執行完 sh ./download.sh，將會載下5個檔案，分別為：
    + best_extractive_model.ckpt
    + best_seq2seq_abstractive_model.ckpt
    + best_seq2seq_attn_model.ckpt
    + embedding_seq2seq.pkl
    + embedding_tag.pkl
2. 接著執行各個預測 shell file，分別為：
    + extractive.sh
    + seq2seq.sh
    + attention.sh
3. 調用方式皆為以下，並請皆輸入"絕對路徑" (參照助教討論區敘述)
```bash=
sh ./<*.sh> <input_file> <output_file>
```
4. 由於 seq2seq 以及 attention 是共用同一個 preprocess code，所以我 ""沒有針對不同的輸入提供不同的 embedded file檔名""，故請務必跑完一輪預測再跑一個新的，前一份生出來的中間檔案 (embedded file) 會被後一份蓋掉！

## 各個file是甚麼

1. extractive.sh
    + 調用 ./src/preprocess_seq_tag.py 生成 embedded file
    + 調用 extractive.py 生成預測
2. seq2seq.sh
    + 調用 ./src/preprocess_seq2seq.py 生成 embedded file
    + 調用 seq2seq.py 生成預測
3. attention.sh
    + 調用 ./src/preprocess_seq2seq.py 生成 embedded file
    + 調用 attention.py 生成預測
4. extractive_nb.ipynb
    + extractive model 訓練用，以及 report 4 hist 生成，一路按到底，配合更改存放 model 的路徑理論上就會生成 report 4 hist. 
5. seq2seq_nb.ipynb
    + seq2seq model 訓練用
6. seq2seq_attn.ipynb
    + seq2seq + attention model 訓練用
7. report.pdf
    + 就是個報告檔案
8. README.md
    + 就是個readme檔案，你正在看的^^