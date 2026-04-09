from sae.train import train_sae

# Run a quick test with minimal settings
print("Testing SAE training with minimal settings...")
sae = train_sae(
    layer_name="transformer.h.6",
    epochs=5,
    buffer_size=2**18,
    batch_size=2**12,
    lr=1e-4,
    l1_coef=1e-3,
)

print("\\n=== Training completed successfully! ===")
print(f"SAE input_dim: {sae.input_dim}")
print(f"SAE hidden_dim: {sae.hidden_dim}")
print(f"Encoder weight shape: {sae.W_enc.weight.shape}")
print(f"Decoder weight shape: {sae.W_dec.weight.shape}")
