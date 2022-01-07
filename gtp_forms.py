import torch
from transformers import GPT2Tokenizer,GPT2LMHeadModel

gpuN=None
if gpuN is not None:
    device = torch.device('cuda:' + str(gpuN))
else:
    device = torch.device('cpu')
prompt_text=['This form is to be filled out regarding Cleopatra','Information regarding Cleopatra']

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

inputs = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
inputs = inputs.to(device)

output_sequences = model.generate(
        input_ids=inputs,
        max_length=args.length + len(encoded_prompt[0]),
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        num_return_sequences=args.num_return_sequences,
    )
#inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
#outputs = model(**inputs, labels=inputs["input_ids"])
#logits = outputs.logits

# Remove the batch dimension when returning multiple sequences
if len(output_sequences.shape) > 2:
    output_sequences.squeeze_()

generated_sequences = []

for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
    print(f"=== GENERATED SEQUENCE {generated_sequence_idx + 1} ===")
    generated_sequence = generated_sequence.tolist()

    # Decode text
    text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

    # Remove all text after the stop token
    text = text[: text.find(args.stop_token) if args.stop_token else None]

    # Add the prompt at the beginning of the sequence. Remove the excess text that was used for pre-processing
    total_sequence = (
        prompt_text + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)) :]
    )

    generated_sequences.append(total_sequence)
    print(total_sequence)
