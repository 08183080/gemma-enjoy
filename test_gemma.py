from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("C:\\gemma-2b")
model = AutoModelForCausalLM.from_pretrained("C:\\gemma-2b")

inpu_1= "Who are you?"
input_2 = 'please write a poem about love'
input_3 = "你是谁呀? 回答我"

input_ids = tokenizer(input_2, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))