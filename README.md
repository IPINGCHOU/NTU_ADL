# NTU_ADL
108-2 深度學習應用 Applied Deep Learning

## HW1-Summarization

Input: Text <br />
The Grade I listed Harnham Gate was hit by a white van that smashed into the structure at about 02:00 BST.\nA 51-year-old man, from West Dean, has been arrested on suspicion of failing to stop, criminal damage and driving with excess alcohol, police said.\nWiltshire Police said the man remains in police custody and they have asked for witnesses to contact them.
<br />
- - -
Output: Summary <br />
A seven-hundred-year old oak gate at Salisbury Cathedral has been demolished by a drink driver.

### Task
1. Extractive model : Extract the sentences in the text as summary. A binary question (to take or not to take the sentences)
2. Abstractive model : Generate the summary by given text and corresponding summary. A seq2seq generation task.


* Train an extractive model and pass baseline: (Extractive model)
  * Rouge-1: 18.5, Rouge-2: 2.6, Rouge-L: 12.3
* Train a Seq2Seq model and pass baseline: (Abstractive model)
  * Rouge-1: 15.0, Rouge-2: 1.8, Rouge-L: 13.0
* Train a Seq2Seq+Attention model and pass baseline: (Abstractive model)
  * Rouge-1: 25.0, Rouge-2: 5.0, Rouge-L: 20.0

All baseline passed with valid dataset.

