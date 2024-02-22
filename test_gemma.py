from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("C:\\gemma-2b")
model = AutoModelForCausalLM.from_pretrained("C:\\gemma-2b")

input_text = "Who are you?"
input_chi = "你是谁呀?"
input_ids = tokenizer(input_chi, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))

print(outputs)