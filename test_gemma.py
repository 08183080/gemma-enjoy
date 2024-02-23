from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("C:\\gemma-2b")
model = AutoModelForCausalLM.from_pretrained("C:\\gemma-2b")

input_1= "Who are you?"
input_2 = 'please write a poem about love'
input_3 = "你是谁呀? 回答我"
input_4 = 'how to become a life winner?'
input_5 = 'how ca i become a successful python coder, plese give me some tips'

input_ids = tokenizer(input_5, return_tensors="pt")

outputs = model.generate(**input_ids, max_new_tokens=300)

print(tokenizer.decode(outputs[0]))