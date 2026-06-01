from transformers import AutoTokenizer, AutoModel

repo = "InstaDeepAI/NTv3_650M_post"
tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
model = AutoModel.from_pretrained(repo, trust_remote_code=True)

# Prepare inputs
batch = tokenizer(["ATCGNATCG", "ACGT"], add_special_tokens=False, padding=True, pad_to_multiple_of=128, return_tensors="pt")

# Species tokens
species = ['human', 'mouse']
species_ids = model.encode_species(species)

# Forward pass
out = model(
    input_ids=batch["input_ids"],
    species_ids=species_ids,
)

print(out.logits.shape)                 # MLM logits: (B, L, V = 11)
print(out.bigwig_tracks_logits.shape)   # BigWig predictions
print(out.bed_tracks_logits.shape)         # Bed track predictions
