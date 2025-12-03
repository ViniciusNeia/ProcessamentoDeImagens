import splitfolders

input_path = '../dataset/potato_original'
output_path = '../datasets/potato-dataset'

splitfolders.ratio(
    input=input_path,
    output=output_path,
    seed=42,
    ratio=(0.8, 0.1, 0.1),  # Treino, Validação, Teste
    group_prefix=None,
    move=False
)
