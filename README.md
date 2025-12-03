**Analise de Saude de Batatas**

Equipe: Vinicius Gonçalves Néia (RA:2454467)

Descrição:
- **Resumo**: Projeto para treino e avaliação de um classificador CNN (ResNet v2) para detecção de doenças em batatas usando o dataset "Potato".

**Estrutura Básica do Projeto**
- **`datasets/`**: local onde o dataset deve ser colocado. Espera a estrutura `potato-dataset/{train,val,test}/{classes}`.
- **`src/`**: código principal; contém os scripts `train.py`, `test.py`, `model.py` e utilitários.
- **`modelos/`**: local onde o modelo treinado é salvo (ex.: `modelo_resnet_v2.keras`).
- **`resultados/`**: saída de métricas, matrizes de confusão e histórico de treino.

**Links**
- **Dataset (Kaggle)**: https://www.kaggle.com/datasets/faysalmiah1721758/potato-dataset
- **Repositório base / referência**: https://github.com/ViniciusNeia/ProcessamentoDeImagens

**Requisitos**
- **Python**: 3.10+ (recomendado 3.10–3.11)
- **Dependências**: veja `src/requirements.txt`.

**Como configurar o ambiente (Windows PowerShell)**
1. No nível do projeto (`ProjetoFinalImagens`), crie e ative um ambiente virtual(É necessario python 3.10+):

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instale as dependências:

```
pip install -r src\requirements.txt
```

3. Coloque o dataset na pasta `datasets` com o nome `potato-dataset` (caminho esperado: `ProjetoFinalImagens/datasets/potato-dataset`).

**Como rodar**
- **Treinamento**: entre na pasta `src` e execute:

```
cd src
python train.py
```

O `train.py` usa por padrão os caminhos relativos:
- `DATASET` = `../datasets/potato-dataset`
- `MODELO_PATH` = `../modelos/modelo_resnet_v2.keras`
- `RESULTADOS` = `../resultados/`

Ao final do treino o modelo será salvo em `modelos/` e o histórico/métricas em `resultados/`.

- **Avaliação / Teste**: com o modelo treinado em `modelos/`, execute:

```
cd src
python test.py
```

O `test.py` carrega o modelo de `../modelos/modelo_resnet_v2.keras` e gera a matriz de confusão e métricas em `../resultados/`.

