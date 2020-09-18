# Neural Question Generation

This is module acts as the second part of the chain, taking input from the summarizer and generating questions for the question answering unit.


## Text Preprocessing
The dataset used is [SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/)
It is meant for contextual question answering hence is optimal considering it has both answer and context in the supervised (train) set.
The train set has around 90k examples. For initial experimentation 20k were used.
spaCy has been used for tokenisation which apart from tokenisation also lemmatises the words. Punctuations are removed and words are converted to lower case.
For loading the data and converting to batches, torchtext is used. The iterator used is BucketIterator which can sort the input on some paramter, thereby reducing the amount of padding required. In this case
since the context is the variable that differs most in length, it is used as the key.

## Model Description
The model is an encoder decoder based architecture.
It takes the concatenated version of context and answer as input to the encoder and outputs the question.
![](https://github.com/dsgiitr/IEQA/blob/master/Question_Generation/utils/images/encoder_decoder.jpeg)

## Encoder
The Encoder is a bidirectional RNN, with pretrained [fastText (300d)](https://fasttext.cc/) word embeddings
Output of both forward and backward layers are concatenated.

## Decoder
The decoder is a RNN which takes previous hidden state from context vector and outputs one the probability distribution of the words one at a
time. Instead of just passing the context vector as the first hidden state, it is also passed in each time step to the linear layer which produces the probability distribution.
To ensure that errors in predicting previous words dont go on propogating further, teacheer forcing ratio of 0.5 is used.
![](https://github.com/dsgiitr/IEQA/blob/master/Question_Generation/utils/images/decoder_direct.png)

## Results
#### RNN with no direct connections and no word embeddings
![](https://github.com/dsgiitr/IEQA/blob/master/Question_Generation/utils/images/results_1.png)

### Bidirectional RNN Encoder with direcct connection decoder
![](https://github.com/dsgiitr/IEQA/blob/master/Question_Generation/utils/images/results_2.png)

### Word Embedding (fasText 300d)
![](https://github.com/dsgiitr/IEQA/blob/master/Question_Generation/utils/images/results_3.png)

The one with the word embeddings works best, but has to be trained for more epochs.
### Example of one prediction

In the 4th Epoch
- **Context**- Greece: On March 24, 2008, the Olympic Flame was ignited at Olympia, Greece, site of the ancient Olympic Games. The actress Maria Nafpliotou, in the role of a High Priestess, ignited the torch of the first torchbearer, a silver medalist of the 2004 Summer Olympics in taekwondo Alexandros Nikolaidis from Greece, who handed the flame over to the second torchbearer, Olympic champion in women's breaststroke Luo Xuejuan from China. Following the recent unrest in Tibet, three members of Reporters Without Borders, including Robert M├⌐nard, breached security and attempted to disrupt a speech by Liu Qi, the head of Beijing's Olympic organising committee during the torch lighting ceremony in Olympia, Greece. The People's Republic of China called this a "disgraceful" attempt to sabotage the Olympics. On March 30, 2008 in Athens, during ceremonies marking the handing over of the torch from Greek officials to organizers of the Beijing games, demonstrators shouted 'Free Tibet' and unfurled banners; some 10 of the 15 protesters were taken into police detention. After the hand-off, protests continued internationally, with particularly violent confrontations with police in Nepal.
Answer-Alexandros Nikolaidis

- **Predicted Question** - ['what be the name of the to that the]
- **Actual Question** - ['what be the name of the person who hand off the torch to the torchbearer in the united states 2008 olympic relay]

The model is able to pickup the idea of the question and the fact that it must be about a name within just a few epochs. Although it is not giving grammatically correct outputs.

This work is inspired by [Learning to Ask: Neural Question Generation for Reading Comprehension by Xinya Du et.al.](https://arxiv.org/abs/1705.00106)

## Future Work
1. Use seperate encoder for answer and context and apply attention of answer output with hidden states of context.
2. [Byte pair encoding](https://arxiv.org/abs/1508.07909)
3. Rule based priors for better grammatical questions
4. [fastText](https://arxiv.org/abs/1607.01759) model for paragraph to speed up encoding process
