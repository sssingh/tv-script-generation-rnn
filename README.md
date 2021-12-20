# TV Script Generation using Recurrent Neural Network (RNN)

<img src=assets/title_image.png width=800>

### Train a RNN based Neural Network to generate fake TV/film script 


## Table of Contents

- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [Solution Approach](#solution-approach)
- [How To Use](#how-to-use)
- [Credits](#credits)
- [License](#license)
- [Author Info](#author-info)


## Introduction
In this project, we'll train a RNN based neural-network to automatically generate our own [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) _fake_ TV script. We'll make use a small subset of the Seinfeld TV sit-com's script from 9 of its seasons to train the our network. The trained network will then generate a new but _fake_ TV script, based on patterns it learned from the training data. RNNs are perfectly suited for sequence problems such as these since they take advantage of the underlying structure of the data, namely, the order of the data points.

## Objective
To build a RNN based Neural Network that'd accept a text corpus (TV/film script, book etc), learn from it and then given a seed `prime-word` it'd generate a _fake_ text snippet that'd look like if this generated text actually came from the original script/book. 


---
## Dataset
- The dataset used in this project is a subset of original [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) on Kaggle.
- The dataset to be used is provided as part of this repo and its kept in `data/Seinfeld_Scripts.txt` as a plain text file.

---
## Solution Approach
### Load and Explore Data
- load the Seinfeld TV script `Seinfeld_Scripts.txt` as a text blob
- Once data is loaded, we play around and explore data to view different parts of the script. This gives us a sense/structure of the data we'll be working with. For example, that it is all lowercase text, and each new line of dialogue is separated by a newline character `\n`. We could extract information such as...

    **Dataset Stats**<br>
    Roughly the number of unique words: 46367<br>
    Number of lines: 109233<br>
    Average number of words in each line: 5.544240293684143<br>
    The lines 0 to 10:<br>
    >jerry: do you know what this is all about? do you know, why were here? to be out, this is out...and out is one of the single most enjoyable experiences of life. people...did you ever hear people talking about we should go out? this is what theyre talking about...this whole thing, were all out now, no one is home. not one person here is home, were all out! there are people trying to find us, they dont know where we are. (on an imaginary phone) did you ring?, i cant find him. where did he go? he didnt tell me where he was going. he must have gone out. you wanna go out you get ready, you pick out the clothes, right? you take the shower, you get all ready, get the cash, get your friends, the car, the spot, the reservation...then youre standing around, what do you do? you go we gotta be getting back. once youre out, you wanna get back! you wanna go to sleep, you wanna get up, you wanna go out again tomorrow, right? where ever you are in life, its my feeling, youve gotta go.<br>
    >jerry: (pointing at georges shirt) see, to me, that button is in the worst possible spot. the second button literally makes or breaks the shirt, look at it. its too high! its in no-mans-land. you look like you live with your mother. <br>
    >george: are you through? <br>
    >jerry: you do of course try on, when you buy? <br>
    >george: yes, it was purple, i liked it, i dont actually recall considering the buttons. 

### Pre-process Data
We'd need to prepare the textual data for modeling i.e. pre-process data so that it can be fed to a Neural Network as `numbers`.

- We define a `lookup-table` (dictionary) for `punctuation-symbols` as shown below and then replace each occurrence of punctuation with its corresponding symbol we defined. <br><br>
  ```python
     punct = {'.': '||Period||',
              ',': '||Comma||',
              '"': '||Quotation_Mark||',
              ';': '||Semicolon||',
              '!': '||Exclamation_mark||',
              '?': '||Question_mark||',
              '(': '||Left_Parentheses||',
              ')': '||Right_Parentheses||',
              '-': '||Dash||',
              '\n': '||Return||'    
    }
  ```
- The text corpus is then converted into `lower-case` letters
- We then look through the whole text corpus and generate two mappings...
  1. `vocab_to_int` : This maps each _unique_ `word` in the text to an _unique_ `integer` (value). This map will be used to encode the text into numbers so that we can feed to our network.
  2. `int_to_vocab` : This is reverse of the above map where each unique `integer` (key) is mapped to an _unique_ `word` (value). This map will be used to convert the model generated output which are numbers back to text so that it can be presented back to user as text.
- Using the `vocab_to_int` map we encode our text corpus into a huge list of numbers.
- Since data pre-processing done above can take quite a while to complete, as a final step, we serialize (`pickle`) all three maps created above and encoded text as python objects in `preprocess.p` file. We do this so that if we need to run our notebook from start again then we do not need to run pre-processing again , it'd save us lot of time.

### Prepare Data Batches
The problem at hand mandates our network to learn from sequential data (i.e. data-elements in a particular order, words of a sentence in case of a TV/Film script) and then be able to produce another sequential data (fake TV/Film script) that'd appear to have come from original sequential data corpus. In other words our objective is to generate `meaningful` text. Our network will consume a starting word (`prime word`) and then it'd try to predict next most-likely word based on the prime-word provided. It'd then try to predict next word based on previously predicted words and so on. Unlike a conventional dataset where we have a well defined set of `input` features (independent variables) and `target` feature (dependent variable) for models to learn from, the textual data does not offer the obvious input and target segregation. Sequential textual data needs to be broken down to produce `input` features (these would be sequence of words in our case) and `target` feature (this is the next word that appears next in the given text). The inputs and targets are then paired to create a single data-point (data sample) and data-points are then batched together to make a training batch of a specified size that can be fed to a neural-network.

We have defined a function `batch_data(words, sequence_length, batch_size)` that will perform the data batching operation. Function would split the encoded input-text (`words`) into sequences of words, each sequence will have equal number of words. First sequence will start from _first-word_ up to to `sequence_length` next words in the input text, where each word in the sequence can be considered as _independent-variable_. This sequence will then paired with _next-word_ in the input text as _dependent-variable_ (target). The paired sequence and its target is a single data-point. This process is repeated but this time sequence starts from `second-word` in the input text and so on. Finally data-points are packed into batches of given batch_size using `TensorDataset` and `DataLoader` objects.

For example, say we have these as input:
```
words = [1, 2, 3, 4, 5, 6, 7]
sequence_length = 4
```

The first-sequence would contain the values:
```
[1, 2, 3, 4] # independent-variable (input features)
```
And the corresponding target` should just be the next word:
```
5 # dependent-variable (target)
```
Now the second-sequence would contain the values (note that its starting from second word (i.e. 2) this time):
```
[2, 3, 4, 5]  # independent-variable (input features)
6             # dependent-variable (target)
```

For example lets say we have the encoded input text as numbers from [0...52], a call to `batch_data(test_text, sequence_length=5, batch_size=10)` will produce the 2 batches. The first batch (maximum 10 sequences of maximum 5 words each) as shown below...

```python
# independent-variable (input features)
tensor([[ 0,  1,  2,  3,  4],
        [ 5,  6,  7,  8,  9],
        [10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19],
        [20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29],
        [30, 31, 32, 33, 34],
        [35, 36, 37, 38, 39],
        [40, 41, 42, 43, 44],
        [45, 46, 47, 48, 49]])

# dependent-variable (target)
tensor([ 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
```

Note that above `target` is always the next word after the sequence (5 for first sequence, 10 for next and so on)


Since there are more than 50 words the second batch will be prepared with left over words, as shown below...

```python
# independent-variable (input features)
tensor([[21387, 21387,    50,    51,    52]])

# independent-variable (input features)
tensor([0])
```

Couple of important things to note... 
(1) Since there are only 3 words left and they are consumed by the sequence the target is taken as the very first word of the input text (0 in this case)
(2) The first two numbers `21387` are the special padding character used to pad the small sequence to make them of equal size (5 in this case)

### Define The Network
- Since we are dealing with sequential data we'll use a RNN based neural-network that are best suited for such task. RNN is well suited for sequential data because in case of sequences the order in which the data-element is appearing is very important. RNN cell keep track of the cell state (`hidden` state) of *previous* `time-step` along with input from *current* `time-step` to produce an output (which is again input to next time-step as previous hidden state). A typical high-level structure of RNN cell is shown below...

![](assets/typical_rnn_cell.png)

A very important point to note here is that even though in right-side diagram two cells are shown one after the other, in fact its the same cell shown as _unrolled_ just for clarity. Same cell consumes its own previous `hidden state` from previous time step and with current time-step input and produces `hidden state` for next time step. For example each word in a sentence (sequence) can be considered as a time step and they will be processed one at a time, one time step output (hidden state) feeding into the next one, i.e. a `loop-back` type of mechanism. This is the behavior of RNN cell that allows them to account for dependency among the input and able to extract `semantic` meaning out of them.

Mathematically at time step `t`, the hidden output of RNN is computed as follows...

![](assets/rnn_formula.png)

As shown above, the `tanh` (Hyperbolic Tangent) function is used as the _activation_ function here. Its very typical of RNN networks to use `tanh` instead of `ReLU` activation function used by other types of neural-networks. This is done because `ReLU` has no upper-bound, however `tanh` maps the feature space to the interval (-1, 1). This ensures that, at every step of the sequence, the hidden state is always between -1 and 1. Given that we have only one linear layer to transform the hidden state, regardless of which step of the sequence it is being used in, it is definitely convenient to have its values within a predictable range.

An RNN cell structure at lower level (neuron level) is shown below...

![](assets/rnn_cell_internals.png)

The number of hidden-neurons are the `hyperparameter` of the network, which is chosen to be 2 in the above diagram (hence two blue neuron in hidden state section). The number of neuron in input section will be automatically chosen to be _exactly_ same as that of hidden-neuron, this is because both transformed input and hidden state needs to be added together hence their shape must match. However, the _dimension_ of the  _input_ features (words in our case) could be anything, This is again another hyperparameter of the network. In above diagram its shown as 2 dimensional (x0 and x1) vector. In this project we will use an `Embedding` layer that'd transform each word that we have encoded as _single_ unique number to its multi-dimensional vector re-presentation.

- In practice, the plain vanilla RNN shown above is hardly used now days. RNN gives equal importance to both the previous-hidden-state and the current-input. What if previous-hidden-state contains more information than the new-hidden-state or current-input adds more information than previous-hidden-state? There is no mechanism in vanilla RNNs to assign weights (how much to keep or ignore) to previous-hidden-state, new-hidden-state and current-input. This is where the improved versions of RNN such as `LSTM` (Long Short Term Memory) or `GRU` (Gated Recurrent Unit) comes in. They are utilized to build a RNN based neural net now days. In this project we are going to use GRU cell because its slightly light weight compare to LSTM cell and trains bit faster. If LSTM is used then its possible to get better results than what we are producing with GRU  but it might take much longer to train. A typical GRU cell structure is shown below...

![](assets/gru_cell_internals.png)


![](assets/gru_formula.png)

- We have defined the network as a subclass of `nn.Module` Pytorch class.
- The network architecture is shown below...

![](assets/rnn_network.png)

- The very first layer of the network is an `Embedding` layer which takes input vector of size equal to our vocabulary (number of unique words in our training text corpus) and outputs a vector representation of each of the words in the vocabulary. The size of this vector is yet another `hyperparameter` called `embedding dimension` that we define. We have decided to use `300` as the embedding vector size. Note that the vector representation of words is just yet another weight matrix that embedding layer learns during the training. In fact an embedding layer is nothing but a giant lookup table of weights where each row corresponds to an unique word in our vocabulary. We lookup this table for a given word and extract an `embedding dimension` long vector representation of the given word which is then fed to next layer in the network. 
- The next layer is a `GRU` RNN cell that gets word vectors (embedding _dim long) from the embedding layer as input and produces `hidden_dim` long outputs. Here we are using two layers of GRU cell one stacked on top of the other hence `n_layers` parameter is set to 2.
- We are using a 50% `dropout` for GRU cell and 50% dropout between GRU cell and the final fully-connected layer

#### Feed Forward
The shape of input and output produced by each layer of RNN could get very confusing sometimes. Hence it helps if we closely look at the data shape that goes into  a layer and shape of the output that comes out of the layer. Because of some reason (which is beyond my mental ability!) Pytorch RNN cells (RNN, GRU, LSTM) expects data input and produce outputs in shape where `sequence_length` is the first dimension and  `batch_size` is the second. However we are generally used to get `batch_size` as first dimension from Dataloaders. In order to be able to use `batch_size` as first dimension with RRN we need to use `batch_first = True` parameter, you can see this being used with our GRU cell. The `batch_size` make Pytorch RNN cells handle tensors with batch_size as first dimension for inputs and outputs but the hidden-state would still have batch_size as the second dimension. Since we seldom directly deal with hidden-state its not a big issue but its better to be aware of this Pytorch RNN quirk. 

For our network we have selected below hyperparameters to use...
`batch_size = 64`, `sequence_length = 15`, `embedding_dim = 300`, `hidden_dim = 512`, `n_layers = 2`

RNN cells takes two input tensors...
1. The input-sequence with shape `(batch_size, sequence_length, input_dim)` --> (64, 15, 300) (input_dim is the output of embedding_layer i.e. embedding_dim)
2. The hidden-state with shape `(n_layers, batch_size, hidden_dim)` --> (2, 64, 512)

and produces two output tensors...
1. Output states as output (hidden state) for all time steps with shape `(batch_size, sequence_length, hidden_dim)` --> (64, 15, 512)
2. Final hidden-state representation of the full sequence with shape `(n_layers, batch_size, hidden_dim)` --> Note that batch_size as second dim (2, 64, 512) (These would be fed as previous hidden-state step in next cycle)

In `forward` function of the neural-network we have...
* First, Embedding layer that takes word sequence as input in shape (64, 15, 1) and produces output in shape (64, 300)
* Next, the GRU layer takes output of Embedding layer in shape (64, 300) as first input and hidden-state (from previous time-step or initial hidden weights if its a very first time-step) as second input in shape (2, 64, 512). It then produces the `cell-output` in shape (64, 15, 512) and `cell-hidden-state` in shape (2, 64, 512)
* 50% dropout is then applied to GRU cell outputs
* GRU cell output is then flattened to (64 * 15, 512) and fed to the `fully-connected` linear layer
* The output of `fully-connected` layer is re-shaped into (batch_size, sequence_length, self.output_size) --> (64, 15, 512)
* the last time-step output is then extracted, for each batch.  
* Above extracted output and the hidden-state from GRU is then returned to training loop to be used in next iteration of training.

### Train the Network
- For the network training we are using `Adam` optimizer with `0.0007` learning rate and `CrossEntropyLoss` as loss function. The final learning-rate and hyperparameters listed above were selected based on multiple experimentation... 
    - At first kept lr = 0.001 hidden_dim large enough at 256 and embedding dim at 200 with seq_len as 5 but loss did not seem to decrease. 
    - Keeping all param same increased seq_len slowly and found that seq_len = 15 making it go down but stopped decreasing. 
    - Then increased hidden_dim from 128 to 512 gradually and embedding dim from 100 to 300 and observed that loss started to decrease but in the end it fluctuated a lot. 
    - Keeping all other param same tried  various learning-rate and found that 0.0007 is making loss go down consistently.
    - Also, experimented with n_layers --> when n_layers = 3 loss starts to increase in couple of epochs, when n_layers = 1 loss does not decrease and saturated quickly

With above experimentation we finally settled on below hyperparameters...

    sequence_length = 15  
    batch_size = 64
    num_epochs = 10
    learning_rate = 0.0007 
    embedding_dim = 300
    hidden_dim = 512
    n_layers = 2

- Trained the model with above hyperparameters for 10 epochs, achieved a loss ~2.5 and saved/serialized the model as `trained_rnn.pt`

Training for 10 epoch(s)...
Epoch:    1/10    Loss: 5.568603610992431
Epoch:    2/10    Loss: 4.7756297567817185
Epoch:    3/10    Loss: 4.38765674001819
Epoch:    4/10    Loss: 4.071462697695352
Epoch:    5/10    Loss: 3.7681871848932764
Epoch:    6/10    Loss: 3.480992128825932
Epoch:    7/10    Loss: 3.196643516758158
Epoch:    8/10    Loss: 2.948043690559297
Epoch:    9/10    Loss: 2.709813594689795
Epoch:   10/10    Loss: 2.509170540952323
Model Trained and Saved

### Test the Network
- Loaded the serialized model and pre-processed data and then generated a _fake_script of 400 words shown below...

![](assets/test_result.png)

We can easily see that it looks like the _fake_ generated script snippet came from the original TV script however its far from perfect. There are multiple characters that say (somewhat) complete sentences. All sentences do not make sense but it doesn't have to be perfect! It takes quite a while and lot of experimentation and resources to get really good results for such a complex task. We can use smaller vocabulary (and discard uncommon words), or get more data to improve results. However, its still amazing how we can get very trivial network, with not a lot of effort, that can try to mimic a human generated text up to some degree of accuracy.

Experiment with different prime-word and different script length generates some interesting (and funny) text, give it a try!

---
## How To Use
1. Ensure below listed packages are installed
    - `numpy`
    - `pickle`
    - `torch`
2. Download `tv_script_generation.ipynb` jupyter notebook from this repo
3. In order to train the models, its recommended to execute the notebook one cell at a time as some steps might take a while to execute. If a GPU is available (recommended) then it'll use it automatically else it'll fall back to CPU. 
4. A fully trained model `trained_rnn.pt` and pre-processed data (`preprocess.p`) trained on a small subset of Seinfeld script can be downloaded from [here](https://drive.google.com/file/d/1mwyAofz77zb7PIoWlGXArMlLpUrWpWDM/view?usp=sharing) and here[](https://drive.google.com/file/d/1al45-FofQCpnW_fK_vD9pmdK8xqVhKxV/view?usp=sharing) respectively.
5. Note that even though we have used Seinfeld script to train and generate the text, you can use any other text corpus (tv/film script, books) to re-train the model and then generate script. You will need to pre-process the data and serialize the data as `preprocess.p` before training the model though. 
6. Once we have model and pre-processed data, we can start generating the script. The network needs to start with a single prime-word and repeat its predictions until it reaches a set length. We'll be using the `generate` function to do this. It takes a word id to start with, `prime_id`, and generates a set length of text, `predict_len`. Example code for script generation is shown in below code snippet...


```python
    # Load the pre-processed data saved in preprocess.p 
    _, vocab_to_int, int_to_vocab, token_dict = load_preprocess()
    # Load the trained model
    trained_rnn = load_model('trained_rnn')
    # You can choose a different length
    gen_length = 400 
    # You can set the prime word to _any word_ in our dictionary, but it's best to start with a name for generating a TV script.
    prime_word = 'elaine'
    # Generate script and print
    pad_word = SPECIAL_WORDS['PADDING']
    generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
    print(generated_script)
```


---
## Credits
- Dataset used in this project is provided by [Udacity](https://www.udacity.com/)
- Above dataset is a subset taken from Kaggle, [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv)
- GRU, RNN diagrams courtesy [Daniel V. Godoy](https://github.com/dvgodoy)

---
## License

MIT License

Copyright (c) [2021] [Sunil S. Singh]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Author Info

- Twitter - [@sunilssingh6](https://twitter.com/sunilssingh6)
- Linkedin - [Sunil S. Singh](https://linkedin.com/in/sssingh)


---
