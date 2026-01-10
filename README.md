# English to Arabic Machine Translation (Seq2Seq GRU)

This project implements an **English → Arabic neural machine translation system** using a **Sequence-to-Sequence (Seq2Seq)** architecture with **GRU networks** in **TensorFlow / Keras**.

The model is trained on parallel English–Arabic sentence pairs and learns to generate Arabic translations token by token.

---

## 📌 Model Overview

- **Architecture:** Encoder–Decoder (Seq2Seq)
- **Encoder:** GRU
- **Decoder:** GRU + Softmax output
- **Embeddings:** Trainable word embeddings
- **Framework:** TensorFlow / Keras
- **Loss Function:** Sparse Categorical Crossentropy
- **Optimizer:** RMSprop

---

## 🧠 How It Works

1. **Encoder**
   - Takes an English sentence as input.
   - Converts words to embeddings.
   - Produces a final hidden state (thought vector).

2. **Decoder**
   - Uses the encoder’s hidden state as initial state.
   - Generates Arabic words step-by-step.
   - Stops when the `eos` token is predicted.

3. **Special Tokens**
   - `sos` → Start of sentence  
   - `eos` → End of sentence  

---

## 📂 Dataset Format

The dataset file must be a **tab-separated text file**:

```text
English sentence<TAB>Arabic sentence


Example
Hello how are you	مرحبا كيف حالك


Note:
The Arabic sentence is automatically wrapped with sos and eos tokens during preprocessing.

⚙️ Requirements

Install the required libraries:

pip install numpy tensorflow

🚀 Training the Model
1️⃣ Load and Preprocess Data

Tokenization using Tokenizer

Padding sequences to maximum length

Vocabulary size limited to 10,000 words

2️⃣ Train the Model
model.fit(
    [english_sequences, decoder_input_data],
    np.expand_dims(decoder_output_data, -1),
    batch_size=64,
    epochs=10,
    validation_split=0.2,
    callbacks=callbacks
)

3️⃣ Callbacks Used

EarlyStopping (patience = 3)

ModelCheckpoint (saves best model)

🔍 Inference (Translation)

After training, separate encoder and decoder models are created for inference.

Example Usage
input_sentence = "Hello, how are you?"

translated_sentence = translate_to_arabic(
    input_sentence,
    encoder_model,
    decoder_model,
    english_tokenizer,
    arabic_tokenizer,
    max_english_len,
    max_arabic_len
)

print(translated_sentence)

🧪 Example Output
Input: Hello, how are you?
Output: مرحبا كيف حالك


(Output depends on dataset quality and training time.)

🛠 Model Parameters
Parameter	Value
Embedding Size	128
GRU State Size	256
Batch Size	64
Epochs	10
Vocabulary Size	10,000
⚠️ Limitations

No attention mechanism

Fixed vocabulary size

Performance depends heavily on dataset quality

Not suitable for long sentences without attention

🔮 Possible Improvements

Add Attention mechanism

Use Bidirectional GRU

Replace GRU with Transformer

Use Beam Search decoding

Train on larger datasets

👤 Author

Alaa Hassoun
Machine Translation Project – English to Arabic
