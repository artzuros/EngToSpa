# ENG-2-SPA

This is a seq2seq implementation of English to Spanish Translator using *tensorflow*.

## --
* Encoder-Decoder LSTM models with seq2seq architecture can be used to solve many problems where input and output are divided in multiple-steps.
* The same architecture can also be modified to work for other tasks like - Summarizing, Conversational models and **Machine Translation**.
* ## Hyperparameters -->
    * INPUT_COLUMN = 'input'
    * TARGET_COLUMN = 'target'
    * TARGET_FOR_INPUT = 'target_for_input'
    * NUM_SAMPLES = 50000
    * MAX_VOCAB_SIZE = 50000
    * EMBEDDING_DIM = 128
    * HIDDEN_DIM = 1024

    * BATCH_SIZE = 64
    * EPOCHS = 10

    * ATTENTION_FUNC = 'general'

* ## Data Preprocessing
    * THe dataset is preprocessed to be able to work on our architecture and vocabulary.
    * We add a *start* and *end* token to our preprocessing to help our model understand where our text starts and ends.

* ## Tokenizing
    * We tokenize the input and target texts using *fit_on_texts* and *text_to_sequences* methods.
    * We then assign a maximum length for both input and targer texts
    * We create *word-to-index* and *index-to-word* tokens

* ## Padding
    * We pad the input sequences upto the maximum length of input sequences and similarly do the same for output sentences
    * This is done as the LSTM wants our input to be in equal length.

* ## Word Embeddings
    * 