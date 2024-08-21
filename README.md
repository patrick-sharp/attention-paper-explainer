```
python3 -m venv env
source env/bin/activate

pip install -r requirements.txt

total_params = sum(p.numel() for p in model.parameters())
total_params = sum(p.numel() for p in m.parameters())

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in m.parameters() if p.requires_grad)

params = [p for p in model.parameters() if p.requires_grad]

python -i repl.py
rf()

black .
```




## Colab specs - from internet
n1-highmem-2 instance
2vCPU @ 2.2GHz
GPU: 1xTesla K80 , compute 3.7, having 2496 CUDA cores , 12GB GDDR5 VRAM
  (better than 1070 in all listed specs)
13GB RAM
100GB Free Space
idle cut-off 90 minutes
maximum 12 hours

2020 Update:
GPU instance downgraded to 64GB disk space.

## Colab GPU (Tensor T4) specs
320 Turing Tensor Cores
2,560 NVIDIA CUDA® cores
8.1 TFLOPS with Single Precision Performance (FP32)
65 FP16 TFLOPS with Mixed Precision (FP16/FP32)
  this is a weird stat that I don't understand
130 INT8 TOPS INT8 Precision
260 INT4 TOPS INT4 Precision
16 GB GDDR6 memory
320+ GB/s bandwidth
70 watts Power draw (peak)?

## p100 specs
3584 CUDA cores
4.7 teraFLOPS
Single-Precision Performance	9.3 teraFLOPS
Half-Precision Performance	18.7 teraFLOPS
12 or 16GB VRAM

## Bibliography
paper
  https://arxiv.org/pdf/1706.03762.pdf
Jay alammar article
  https://jalammar.github.io/illustrated-transformer/
Umar Jamil
  https://www.youtube.com/watch?v=ISNdQcPhsts
karpathy stanford lecture
  https://www.youtube.com/watch?v=XfpMkf4rD6E
karpathy byte pair encoding (gpt tokenizer) lecture
  https://www.youtube.com/watch?v=zduSFxRajkE
coding lane's adam optimizer explainer
  https://www.youtube.com/watch?v=tuU59-G1PgU
how pytorch's automatic backprop works
  https://pytorch.org/blog/computational-graphs-constructed-in-pytorch/
how to train a huggingface tokenizer
  https://huggingface.co/docs/tokenizers/training_from_memory
dataset
  https://huggingface.co/datasets/wmt14/viewer/de-en
  https://www.statmt.org/wmt14/translation-task.html
    three corpora for de-en:
      europarl-v7.de-en
      news-commentary-v9.de-en.
      commoncrawl.de-en
The annotated transformer
  https://nlp.seas.harvard.edu/annotated-transformer/
Smart batching tutorial
  https://mccormickml.com/2020/07/29/smart-batching-tutorial/#s4-smart-batching
The google paper on beam search that they cite
  https://arxiv.org/pdf/1609.08144
  note that they use a length penalty and a coverage penalty. the attention paper doesn't mention a coverage penalty, just length
  in this beam search paper, the length penalty is called length normalization
  Their equation is under Decoder on page 12 of the pdf

## Useful links
backend specs colab
  https://colab.research.google.com/drive/1_x67fw9y5aBW72a8aGePFLlkPvKLpnBl#scrollTo=QlvkREnwN-RA
pytorch docs
  https://pytorch.org/docs/stable/

## Specific torch docs
https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics
https://pytorch.org/docs/stable/generated/torch.matmul.html


