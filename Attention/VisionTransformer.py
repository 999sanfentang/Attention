import numpy as np


def soft_max(z):
    t = np.exp(z)
    a = np.exp(z) / np.expand_dims(np.sum(t, axis=-1), -1)
    return a


values_length = 3
num_attention_heads = 8
hidden_size = 768
attention_head_size = hidden_size // num_attention_heads

Query = np.random.rand(values_length, hidden_size)
Key = np.random.rand(values_length, hidden_size)
Value = np.random.rand(values_length, hidden_size)
print(np.shape(Query))

Query = np.reshape(
    Query, [values_length, num_attention_heads, attention_head_size])
Key = np.reshape(
    Key, [values_length, num_attention_heads, attention_head_size])
Value = np.reshape(
    Value, [values_length, num_attention_heads, attention_head_size])
print(np.shape(Query))

Query = np.transpose(Query, [1, 0, 2])
Key = np.transpose(Key, [1, 0, 2])
Value = np.transpose(Value, [1, 0, 2])
print(np.shape(Query))
print(np.shape(Query[0]))
print("----------------")

scores = Query @ np.transpose(Key, [0, 2, 1])
print(np.shape(scores))
scores = soft_max(scores)
print(np.shape(scores))
out = scores @ Value
print(np.shape(out))
out = np.transpose(out, [1, 0, 2])
print(np.shape(out))
out = np.reshape(out, [values_length, 768])
print(np.shape(out))
