![[Pasted image 20230628002107.png]]
Guardar encodings de fechas de manera lineal es poco eficiente, así que se desea observar le problema como oscilaciones en un espacio latente.

![[Pasted image 20230628001902.png]]

![[Pasted image 20230628001909.png]]

Una #guia muy completa sobre la arquitectura
https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

Una #guia para implementarla
https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/#:~:text=The%20scheme%20for%20positional%20encoding,way%20of%20encoding%20each%20position.

Una #guia para entenderla
https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3

### Implementación

```python
#### TensorFlow only version ####

def positional_encoding(max_position, d_model, min_freq=1e-4):

	position = tf.range(max_position, dtype=tf.float32)
	
	mask = tf.range(d_model)
	
	sin_mask = tf.cast(mask%2, tf.float32)
	
	cos_mask = 1-sin_mask
	
	exponent = 2*(mask//2)
	
	exponent = tf.cast(exponent, tf.float32)/tf.cast(d_model, tf.float32)
	
	freqs = min_freq**exponent
	
	angles = tf.einsum('i,j->ij', position, freqs)
	
	pos_enc = tf.math.cos(angles)*cos_mask + tf.math.sin(angles)*sin_mask
	
	return pos_enc

#### Numpy version ####

def positional_encoding(max_position, d_model, min_freq=1e-4):

	position = np.arange(max_position)
	
	freqs = min_freq**(2*(np.arange(d_model)//2)/d_model)
	
	pos_enc = position.reshape(-1,1)*freqs.reshape(1,-1)
	
	pos_enc[:, ::2] = np.cos(pos_enc[:, ::2])
	
	pos_enc[:, 1::2] = np.sin(pos_enc[:, 1::2])
	
	return pos_enc

### Plotting ####

d_model = 128

max_pos = 256

mat = positional_encoding(max_pos, d_model)

plt.pcolormesh(mat, cmap='copper')

plt.xlabel('Depth')

plt.xlim((0, d_model))

plt.ylabel('Position')![[Pasted image 20230628002059.png]]



plt.title("PE matrix heat map")

plt.colorbar()

plt.show()
```




___


## What is positional encoding and Why do we need it in the first place?

Position and order of words are the essential parts of any language. They define the grammar and thus the actual semantics of a sentence. Recurrent Neural Networks (RNNs) inherently take the order of word into account; They parse a sentence word by word in a sequential manner. This will integrate the words’ order in the backbone of RNNs.

But the Transformer architecture ditched the recurrence mechanism in favor of multi-head self-attention mechanism. Avoiding the RNNs’ method of recurrence will result in massive speed-up in the training time. And theoretically, it can capture longer dependencies in a sentence.

As each word in a sentence simultaneously flows through the Transformer’s encoder/decoder stack, The model itself doesn’t have any sense of position/order for each word. Consequently, there’s still the need for a way to incorporate the order of the words into our model.

One possible solution to give the model some sense of order is to add a piece of information to each word about its position in the sentence. We call this “piece of information”, the positional encoding.

The first idea that might come to mind is to assign a number to each time-step within the [0, 1] range in which 0 means the first word and 1 is the last time-step. Could you figure out what kind of issues it would cause? One of the problems it will introduce is that you can’t figure out how many words are present within a specific range. In other words, time-step delta doesn’t have consistent meaning across different sentences.

Another idea is to assign a number to each time-step linearly. That is, the first word is given “1”, the second word is given “2”, and so on. The problem with this approach is that not only the values could get quite large, but also our model can face sentences longer than the ones in training. In addition, our model may not see any sample with one specific length which would hurt generalization of our model.

Ideally, the following criteria should be satisfied:

-   It should output a unique encoding for each time-step (word’s position in a sentence)
-   Distance between any two time-steps should be consistent across sentences with different lengths.
-   Our model should generalize to longer sentences without any efforts. Its values should be bounded.
-   It must be deterministic.
