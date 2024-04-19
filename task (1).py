{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "07dde099",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "14d1babe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>externalStatus</th>\n",
       "      <th>internalStatus</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>PORT OUT</td>\n",
       "      <td>Port Out</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>TERMINAL IN</td>\n",
       "      <td>Inbound Terminal</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>PORT IN</td>\n",
       "      <td>Port In</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Vessel departure from first POL (Vessel name :...</td>\n",
       "      <td>Departure</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Vessel arrival at final POD (Vessel name : TIA...</td>\n",
       "      <td>Arrival</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                      externalStatus    internalStatus\n",
       "0                                           PORT OUT          Port Out\n",
       "1                                        TERMINAL IN  Inbound Terminal\n",
       "2                                            PORT IN           Port In\n",
       "3  Vessel departure from first POL (Vessel name :...         Departure\n",
       "4  Vessel arrival at final POD (Vessel name : TIA...           Arrival"
      ]
     },
     "execution_count": 104,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#loading the internal status dataset to a pandas Dataframe\n",
    "df=pd.read_json(\"dataset.json\")\n",
    "#printing the first 5 rows of the dataset\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "f8949d93",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1222 entries, 0 to 1221\n",
      "Data columns (total 2 columns):\n",
      " #   Column          Non-Null Count  Dtype \n",
      "---  ------          --------------  ----- \n",
      " 0   externalStatus  1222 non-null   object\n",
      " 1   internalStatus  1222 non-null   object\n",
      "dtypes: object(2)\n",
      "memory usage: 19.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "ea09a10b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(1222, 2)"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#printing the  rows and columns of the dataset\n",
    "df.shape\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 107,
   "id": "d77535f6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>count</th>\n",
       "      <th>unique</th>\n",
       "      <th>top</th>\n",
       "      <th>freq</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>externalStatus</th>\n",
       "      <td>1222</td>\n",
       "      <td>108</td>\n",
       "      <td>Gate out</td>\n",
       "      <td>144</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>internalStatus</th>\n",
       "      <td>1222</td>\n",
       "      <td>15</td>\n",
       "      <td>Loaded on Vessel</td>\n",
       "      <td>331</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               count unique               top freq\n",
       "externalStatus  1222    108          Gate out  144\n",
       "internalStatus  1222     15  Loaded on Vessel  331"
      ]
     },
     "execution_count": 107,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Transpose the summary statistics DataFrame\n",
    "df.describe().T\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 108,
   "id": "5f6f74b1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "externalStatus    0\n",
       "internalStatus    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 108,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Count the number of missing values in each column\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "id": "aaeb3d87",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic text cleaning\n",
    "df['externalStatus'] = df['externalStatus'].str.lower().str.replace('[^\\w\\s]', '')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "id": "bf9d94ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encoding labels\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "label_encoder = LabelEncoder()\n",
    "df['internalStatus'] = label_encoder.fit_transform(df['internalStatus'])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "id": "56fe506b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Tokenize text\n",
    "\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "tokenizer = Tokenizer(num_words=10000, oov_token=\"<OOV>\")\n",
    "tokenizer.fit_on_texts(df['externalStatus'])\n",
    "sequences = tokenizer.texts_to_sequences(df['externalStatus'])\n",
    "padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "id": "8234ffc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Splitting the dataset\n",
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(padded_sequences, df['internalStatus'], test_size=0.1, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "id": "114b77aa",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential_15\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " embedding_8 (Embedding)     (None, 50, 64)            640000    \n",
      "                                                                 \n",
      " conv1d (Conv1D)             (None, 48, 32)            6176      \n",
      "                                                                 \n",
      " global_max_pooling1d (Glob  (None, 32)                0         \n",
      " alMaxPooling1D)                                                 \n",
      "                                                                 \n",
      " dense_36 (Dense)            (None, 10)                330       \n",
      "                                                                 \n",
      " dense_37 (Dense)            (None, 15)                165       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 646671 (2.47 MB)\n",
      "Trainable params: 646671 (2.47 MB)\n",
      "Non-trainable params: 0 (0.00 Byte)\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "#Model Development \n",
    "import tensorflow as tf\n",
    "\n",
    "model = tf.keras.Sequential([\n",
    "    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=50),\n",
    "    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),\n",
    "    tf.keras.layers.GlobalMaxPooling1D(),\n",
    "    tf.keras.layers.Dense(10, activation='relu'),\n",
    "    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 172,
   "id": "5b1af667",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10\n",
      "35/35 [==============================] - 1s 14ms/step - loss: 0.0011 - accuracy: 1.0000 - val_loss: 0.0462 - val_accuracy: 0.9919\n",
      "Epoch 2/10\n",
      "35/35 [==============================] - 0s 13ms/step - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.0462 - val_accuracy: 0.9919\n",
      "Epoch 3/10\n",
      "35/35 [==============================] - 0s 13ms/step - loss: 9.3919e-04 - accuracy: 1.0000 - val_loss: 0.0461 - val_accuracy: 0.9919\n",
      "Epoch 4/10\n",
      "35/35 [==============================] - 0s 12ms/step - loss: 8.7982e-04 - accuracy: 1.0000 - val_loss: 0.0462 - val_accuracy: 0.9919\n",
      "Epoch 5/10\n",
      "35/35 [==============================] - 0s 13ms/step - loss: 8.2920e-04 - accuracy: 1.0000 - val_loss: 0.0465 - val_accuracy: 0.9919\n",
      "Epoch 6/10\n",
      "35/35 [==============================] - 0s 12ms/step - loss: 7.7812e-04 - accuracy: 1.0000 - val_loss: 0.0465 - val_accuracy: 0.9919\n",
      "Epoch 7/10\n",
      "35/35 [==============================] - 0s 12ms/step - loss: 7.3290e-04 - accuracy: 1.0000 - val_loss: 0.0466 - val_accuracy: 0.9919\n",
      "Epoch 8/10\n",
      "35/35 [==============================] - 0s 13ms/step - loss: 6.8977e-04 - accuracy: 1.0000 - val_loss: 0.0465 - val_accuracy: 0.9919\n",
      "Epoch 9/10\n",
      "35/35 [==============================] - 0s 12ms/step - loss: 6.5124e-04 - accuracy: 1.0000 - val_loss: 0.0467 - val_accuracy: 0.9919\n",
      "Epoch 10/10\n",
      "35/35 [==============================] - 0s 12ms/step - loss: 6.1698e-04 - accuracy: 1.0000 - val_loss: 0.0468 - val_accuracy: 0.9919\n"
     ]
    }
   ],
   "source": [
    "history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "id": "8aaa2d6b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4/4 [==============================] - 0s 7ms/step - loss: 0.0468 - accuracy: 0.9919\n",
      "Test accuracy: 0.9918699264526367\n"
     ]
    }
   ],
   "source": [
    "# Evaluate the model on the test set\n",
    "test_loss, test_accuracy = model.evaluate(X_test, y_test)\n",
    "\n",
    "# Print test accuracy\n",
    "\n",
    "print(\"Test accuracy:\",test_accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "id": "f36a2b58",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "pickle.dump(model,open(\"model_gb.pkl\",\"wb\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7d2bd898",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
