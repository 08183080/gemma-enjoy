from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("C:\\gemma-2b")
model = AutoModelForCausalLM.from_pretrained("C:\\gemma-2b")

input_text = "Who are you?"
input2 = 'Can you write a poem, please'
input_chi = "你是谁呀? 回答我"
input_ids = tokenizer(input2, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))

print(outputs)