# Análise do `notebook.ipynb`

Este documento explica [`notebooks/notebook.ipynb`](../notebooks/notebook.ipynb), que é o notebook de modelagem da base Spaceship Titanic. Ele transforma os dados, constrói uma pipeline de pré-processamento, otimiza hiperparâmetros com Optuna, treina modelos, avalia os resultados por validação cruzada e salva artefatos.

Ele deve ser lido junto com [`eda.md`](./eda.md). A EDA explica por que certas decisões fazem sentido; este documento explica como essas decisões são implementadas.

## Visão Geral

O objetivo é prever `Transported`, uma variável binária:

- `1`: passageiro transportado;
- `0`: passageiro não transportado.

O notebook compara três caminhos:

| Caminho | Modelo usado no código | Hiperparâmetros otimizados |
|---|---|---|
| Caminho A | `DecisionTreeClassifier(random_state=42, **best_params)` | `max_depth`, `min_samples_split` |
| Caminho B | `RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)` | `n_estimators`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| Caminho C | `GaussianNB(**best_params)` | `var_smoothing` |

Todos os modelos passam pela mesma ideia de avaliação: a pipeline inteira é ajustada dentro de cada fold da validação cruzada, as predições out-of-fold são guardadas e as métricas são calculadas de forma padronizada.

Antes de avaliar cada caminho, o notebook procura um arquivo `models/params_path_*.json`. Se ele existir, os melhores parâmetros são carregados dali. Se não existir, o Optuna executa uma busca com 30 tentativas e o melhor conjunto encontrado passa a ser usado no modelo final.

## Relação Com A EDA

O [`eda.ipynb`](./eda.md) mostrou alguns padrões importantes:

- `PassengerId`, `Cabin` e `Name` têm informação composta;
- `HomePlanet` se relaciona com grupo de viagem e sobrenome;
- gastos individuais têm muitos zeros e alguns ausentes;
- `TotalSpend` resume consumo, mas não substitui totalmente os gastos individuais;
- `CryoSleep`, `CabinDeck`, `CabinSide`, `HomePlanet` e `Destination` têm sinal em relação a `Transported`;
- `Age` pode ter efeito não linear, especialmente em crianças pequenas.

O `notebook.ipynb` transforma isso em código: cria atributos, imputa valores ausentes, codifica variáveis categóricas e compara modelos capazes de capturar relações lineares e não lineares. O notebook atual também usa Optuna para escolher hiperparâmetros em vez de depender apenas dos padrões do scikit-learn.

## Setup E Carregamento

O notebook importa bibliotecas para:

- dados: `pandas`, `numpy`;
- gráficos: `matplotlib`, `seaborn`;
- persistência: `joblib`, `json`;
- modelos, pipelines, validação e métricas: `scikit-learn`;
- otimização de hiperparâmetros: `optuna`;
- tempo e timestamps: `datetime`, `perf_counter`.

Os caminhos vêm de `src.roots`:

- `TRAIN_DATA_PATH`;
- `TEST_DATA_PATH`;
- `MODELS_DIR`;
- `REPORTS_DIR`;
- `FIGURES_DIR`.

A função `create_required_directories()` cria as pastas de artefatos quando necessário. Depois, `train.csv` e `test.csv` são carregados. O notebook separa:

- `X`: atributos de entrada, ainda brutos;
- `y`: alvo `Transported`, convertido para inteiro.

Manter `X` bruto é importante porque a engenharia de atributos fica dentro da pipeline. Assim, durante a validação cruzada, cada transformação aprende apenas com o fold de treino.

## Engenharia De Atributos

A classe `SpaceshipFeatureBuilder` abre colunas compostas:

| Origem | Atributos criados | Motivo |
|---|---|---|
| `PassengerId` | `PassengerGroup`, `PassengerNum` | Capturar grupos de viagem. |
| `Cabin` | `CabinDeck`, `CabinNum`, `CabinSide` | Separar convés, número e lado da cabine. |
| `Name` | `FamilyName` | Usar sobrenome como indício familiar. |

Depois, remove:

- `PassengerId`;
- `Cabin`;
- `Name`;
- `PassengerNum`;
- `CabinNum`.

`PassengerNum` e `CabinNum` são derivados, mas não entram no modelo atual. Já `PassengerGroup`, `CabinDeck`, `CabinSide` e `FamilyName` são mantidos.

No fluxo atual, `SpaceshipFeatureBuilder` não calcula `TotalSpend`. Ele apenas cria atributos derivados de identificadores compostos; `TotalSpend` é criado depois da imputação dos gastos individuais.

## Imputação De `HomePlanet`

O notebook preenche `HomePlanet` em etapas.

Primeiro, `HomePlanetRuleImputer` usa regras determinísticas:

- se passageiros do mesmo `PassengerGroup` têm `HomePlanet` conhecido, usa a moda do grupo;
- se ainda faltar valor, usa a moda do `FamilyName`.

Essa decisão vem diretamente da EDA, que mostrou associação entre grupo, sobrenome e planeta de origem.

Se ainda faltar valor depois dessas regras, o mesmo transformador usa o valor mais frequente de `HomePlanet` observado no dataset de treino do fold.

Esse fallback é usado apenas para valores que as regras não conseguiram preencher.

## Gastos E `TotalSpend`

O notebook trata os gastos antes de criar a variável agregada. A intenção é evitar que `TotalSpend` seja uma soma parcial quando algum gasto individual está ausente.

O fluxo é:

- se todos os gastos individuais estão presentes, `TotalSpend` é a soma desses gastos;
- se algum gasto individual está ausente, os gastos são imputados com `KNNImputer`;
- depois da imputação, `TotalSpend` é calculado como soma dos cinco gastos individuais completos.

Isso é implementado pela classe `SpendKNNImputerAndTotalSpend`.

As colunas tratadas são:

- `RoomService`;
- `FoodCourt`;
- `ShoppingMall`;
- `Spa`;
- `VRDeck`.

Tecnicamente, no `fit`, o transformador aprende um `KNNImputer` usando apenas essas colunas de gasto. No `transform`, ele converte os gastos para numérico, imputa ausências quando necessário e cria:

```python
TotalSpend = RoomService + FoodCourt + ShoppingMall + Spa + VRDeck
```

Com isso, `TotalSpend` fica consistente com os gastos que entram no modelo.

## Pipeline De Pré-Processamento

A pipeline compartilhada segue esta ordem:

- `features`: cria atributos derivados com `SpaceshipFeatureBuilder`;
- `homeplanet_rules`: imputa `HomePlanet` por grupo, sobrenome e moda global do treino;
- `spend_total`: imputa gastos individuais e calcula `TotalSpend`;
- `columns`: aplica o `ColumnTransformer`.

O `ColumnTransformer` separa as colunas em grupos:

| Grupo | Colunas | Tratamento |
|---|---|---|
| Numéricas | `Age`, gastos individuais já imputados, `TotalSpend` já calculado | `KNNImputer` para nulos restantes, principalmente `Age`, e `StandardScaler`. |
| Binárias | `CabinSide`, `CryoSleep`, `VIP` | `OrdinalEncoder`, `KNNImputer` e arredondamento para 0/1. |
| Categóricas | `CabinDeck`, `Destination`, `HomePlanet`, `PassengerGroup`, `FamilyName` | `SimpleImputer` e `OneHotEncoder`. |

O `KNNImputer` das numéricas ainda fica na pipeline porque `Age` pode ter ausências e porque ele funciona como uma camada de segurança. Porém, no caso dos gastos e de `TotalSpend`, a etapa principal já aconteceu antes em `SpendKNNImputerAndTotalSpend`.

Para o Caminho C, existe uma variação mínima dessa pipeline:

```python
gaussian_nb_preprocessing_pipeline = clone(preprocessing_pipeline).set_params(
    columns__cat__encoder__sparse_output=False,
)
```

A única diferença é que o `OneHotEncoder` gera matriz densa. Isso é necessário porque o `GaussianNB` trabalha com arrays densos para estimar médias e variâncias por classe.

## Validação Cruzada

O notebook usa:

```python
StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
```

Validação cruzada divide o treino em várias partes chamadas folds. Com 10 folds, o processo funciona assim:

- os dados são divididos em 10 partes;
- em cada rodada, 9 partes são usadas para treino e 1 para validação;
- o processo se repete até cada parte ter sido validação uma vez;
- as métricas são calculadas em cada fold;
- depois, o notebook calcula médias, desvios e métricas out-of-fold.

O termo "estratificada" significa que cada fold tenta preservar a proporção original das classes `Transported=0` e `Transported=1`. Isso é importante mesmo com alvo quase balanceado, porque evita folds acidentalmente enviesados.

A pipeline completa é clonada em cada fold. Isso evita vazamento de dados: imputadores, encoders, scaler e modelo são ajustados apenas no treino daquele fold.

## Como O Optuna Entra No Fluxo

Optuna é a biblioteca usada para automatizar a busca de hiperparâmetros. Em vez de escolher manualmente valores como profundidade da árvore ou número de árvores da floresta, o notebook define um espaço de busca e deixa o Optuna testar combinações.

Cada caminho segue a mesma estrutura:

```python
def objective(trial):
    params = {
        ...
    }
    model = ModelClass(**params)
    pipeline = make_model_pipeline(model)
    return cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    ).mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
best_params = study.best_params
```

Os elementos principais são:

- `study`: objeto que guarda o histórico da busca;
- `trial`: uma tentativa individual, com uma combinação de hiperparâmetros;
- `suggest_int`, `suggest_float` e `suggest_categorical`: funções que definem como sortear/testar valores;
- `objective`: função que recebe um `trial`, treina/valida uma pipeline e retorna a métrica a maximizar;
- `direction="maximize"`: indica que valores maiores de ROC AUC são melhores;
- `n_trials=30`: limita a busca a 30 combinações por caminho.

O Optuna não faz uma grade exaustiva. Ele testa combinações, observa quais tiveram melhor resultado e usa esse histórico para escolher próximas tentativas de forma mais inteligente. Como o notebook chama `optuna.create_study()` sem passar um sampler explícito, ele usa o sampler padrão do Optuna para otimização de objetivo único, que na versão usada normalmente é o `TPESampler`.

## Como Funciona O `TPESampler`

O `TPESampler` é baseado em TPE, ou Tree-structured Parzen Estimator. A intuição é simples:

- primeiro, o Optuna executa algumas tentativas para coletar evidências;
- depois, separa tentativas melhores e piores de acordo com a métrica;
- estima distribuições de probabilidade para os valores que aparecem em cada grupo;
- sugere novos hiperparâmetros que tendem a estar mais próximos das tentativas boas do que das ruins.

Na prática, isso costuma explorar o espaço de busca melhor do que testar valores totalmente aleatórios, especialmente quando há poucos trials. Ainda assim, com `n_trials=30`, o resultado é uma boa busca prática, não uma prova de ótimo global.

## Espaços De Busca Do Optuna

O notebook define espaços diferentes para cada caminho:

| Caminho | Hiperparâmetro | Espaço testado | Interpretação |
|---|---|---|---|
| A | `max_depth` | inteiro de 3 a 20 | Limita a profundidade da árvore. |
| A | `min_samples_split` | inteiro de 2 a 50 | Exige mais amostras para dividir um nó. |
| B | `n_estimators` | inteiro de 50 a 300 | Define quantas árvores entram na floresta. |
| B | `min_samples_split` | inteiro de 2 a 50 | Controla quando uma árvore pode dividir um nó. |
| B | `min_samples_leaf` | inteiro de 1 a 20 | Controla o tamanho mínimo das folhas. |
| B | `max_features` | `sqrt`, `log2` ou `None` | Define quantos atributos cada divisão pode considerar. |
| C | `var_smoothing` | `1e-9` a `1e-3`, em escala log | Adiciona suavização às variâncias do `GaussianNB`. |

O objetivo usado nos três caminhos é a média de `roc_auc` em validação cruzada:

```python
cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()
```

Isso significa que o Optuna escolhe os hiperparâmetros que mais aumentam a capacidade do modelo de ranquear corretamente passageiros transportados e não transportados ao longo dos folds.

## Persistência Dos Parâmetros

Antes de rodar a busca, cada caminho verifica:

```python
params_path = MODELS_DIR / f"params_{experiment_id}.json"

if params_path.exists():
    with open(params_path, "r") as f:
        best_params = json.load(f)
else:
    ...
```

Esse mecanismo evita repetir uma busca cara toda vez que o notebook é executado. Depois da avaliação final, `save_experiment_artifacts` recebe `best_params` e salva o JSON correspondente em `models/`.

Na execução atual do repositório, os parâmetros salvos são:

| Caminho | Arquivo | Parâmetros |
|---|---|---|
| A | `models/params_path_a.json` | `max_depth=6`, `min_samples_split=43` |
| B | `models/params_path_b.json` | `n_estimators=216`, `min_samples_split=26`, `min_samples_leaf=11`, `max_features=None` |
| C | `models/params_path_c.json` | `var_smoothing=0.0009815405019274585` |

Um detalhe de reprodutibilidade: os modelos e os folds têm `random_state=42`, mas o sampler do Optuna não recebe uma seed explícita. Por isso, se os arquivos `params_path_*.json` forem apagados e a busca rodar novamente, os melhores parâmetros podem mudar. Com os JSONs preservados, a execução usa os mesmos parâmetros já encontrados.

## Predições Out-Of-Fold

Predições out-of-fold são previsões feitas para registros que não foram vistos no treino do fold correspondente.

O notebook guarda duas séries:

- `oof_pred`: classe prevista, 0 ou 1;
- `oof_score`: probabilidade estimada para a classe 1.

Ao final dos 10 folds, todo registro do treino tem uma previsão feita por um modelo que não treinou nele. Isso permite calcular métricas globais usando todos os registros, mas sem avaliar nenhum registro com um modelo que já o viu no treino.

Como o mesmo conjunto de treino também é usado para escolher hiperparâmetros via Optuna, as métricas são adequadas para comparar os caminhos dentro do projeto, mas não substituem uma avaliação totalmente independente. Para estimar desempenho final com menor viés de seleção, seria possível usar nested cross-validation ou um holdout separado.

## Métricas Calculadas

O notebook calcula:

| Métrica | Como interpretar |
|---|---|
| `accuracy` | Proporção total de acertos. |
| `precision` | Entre os previstos como transportados, quantos realmente foram transportados. |
| `recall` | Entre os realmente transportados, quantos o modelo encontrou. |
| `f1` | Média harmônica entre precisão e recall. |
| `f1_class_0` | F1 tomando a classe 0 como positiva. |
| `f1_class_1` | F1 tomando a classe 1 como positiva. |
| `roc_auc` | Área sob a curva ROC, baseada nos scores probabilísticos. |

As fórmulas centrais são:

- `accuracy = (TP + TN) / (TP + TN + FP + FN)`;
- `precision = TP / (TP + FP)`;
- `recall = TP / (TP + FN)`;
- `F1 = 2 * precision * recall / (precision + recall)`.

Aqui, `TP`, `TN`, `FP` e `FN` significam, respectivamente, verdadeiro positivo, verdadeiro negativo, falso positivo e falso negativo.

## Curva ROC Out-Of-Fold

A curva ROC mostra como o modelo se comporta quando variamos o limiar usado para converter probabilidade em classe.

O modelo produz um score, no notebook:

```python
predict_proba(X_valid_fold)[:, 1]
```

Esse score é a probabilidade estimada de `Transported=1`. Para desenhar a ROC, o scikit-learn testa vários limiares. Para cada limiar:

- scores acima do limiar viram classe 1;
- scores abaixo do limiar viram classe 0;
- calcula-se a taxa de verdadeiro positivo;
- calcula-se a taxa de falso positivo.

As taxas são:

- `TPR = TP / (TP + FN)`, também chamada de recall ou sensibilidade;
- `FPR = FP / (FP + TN)`.

A curva ROC coloca `FPR` no eixo X e `TPR` no eixo Y. Um classificador aleatório tende a seguir a diagonal. Quanto mais a curva sobe para o canto superior esquerdo, melhor.

No notebook, a ROC é out-of-fold porque usa os scores acumulados de todos os folds. Cada score foi produzido por um modelo que não viu aquele registro no treino do fold. Isso torna a curva mais honesta do que uma ROC calculada no próprio treino.

A AUC é a área sob essa curva. Uma leitura prática:

- `0.5`: comportamento próximo ao aleatório;
- acima de `0.5`: melhor que aleatório;
- próximo de `1.0`: separação muito forte entre classes.

## Artefatos E Funções De Apoio

Antes dos modelos, o notebook define utilitários:

- `make_model_pipeline`: junta pré-processamento e estimador;
- `classification_metrics`: calcula as métricas;
- `build_artifact_files`: padroniza nomes de arquivos;
- `save_experiment_artifacts`: salva modelo, parâmetros, histórico, métricas e ROC.

Cada experimento salva:

- modelo em `models/model_path_*_timestamp.joblib`;
- melhores parâmetros em `models/params_path_*.json`;
- histórico por fold em `reports/history_path_*_timestamp.csv`;
- métricas agregadas em `reports/metrics_path_*_timestamp.csv`;
- pontos da ROC em `reports/figures/roc_path_*_timestamp.csv`;
- imagem da ROC em `reports/figures/roc_path_*_timestamp.png`;
- visualização de árvore para os Caminhos A e B em `reports/figures/tree_path_*_timestamp.svg`.

Os arquivos com timestamp representam execuções específicas. Os arquivos `params_path_*.json` não têm timestamp porque funcionam como cache dos melhores parâmetros para cada caminho.

## Caminho A: Árvore De Decisão

Uma árvore de decisão aprende uma sequência de perguntas sobre os atributos. Cada nó interno contém uma regra, por exemplo "idade menor ou igual a certo valor" ou "categoria pertence a determinado grupo". Cada divisão tenta separar melhor as classes.

No final, cada folha guarda uma distribuição de classes. Para classificar um passageiro, o modelo percorre a árvore da raiz até uma folha e retorna a classe mais provável naquela folha.

## Algoritmo Usado Na Árvore

No scikit-learn, `DecisionTreeClassifier` implementa uma árvore binária no estilo CART, com divisão gulosa. "Gulosa" significa que, em cada nó, o algoritmo escolhe a melhor divisão naquele momento, sem testar todas as árvores futuras possíveis.

O notebook atual não deixa a árvore totalmente nos padrões. Ele usa os parâmetros encontrados pelo Optuna:

```python
DecisionTreeClassifier(
    random_state=42,
    max_depth=6,
    min_samples_split=43,
)
```

Esses valores tornam a árvore mais conservadora do que uma árvore sem limite de profundidade. `max_depth=6` limita o comprimento máximo dos caminhos da raiz até as folhas, e `min_samples_split=43` evita divisões baseadas em poucos registros.

## Algoritmos Possíveis Para Árvores De Decisão

Na família de árvores de decisão existem algoritmos clássicos diferentes:

| Algoritmo | Ideia principal |
|---|---|
| ID3 | Usa ganho de informação, historicamente voltado a atributos categóricos. |
| C4.5 | Evolução do ID3, lida melhor com atributos contínuos e poda. |
| CART | Usa árvores binárias e critérios como Gini ou erro quadrático, dependendo da tarefa. |
| CHAID | Usa testes estatísticos, como qui-quadrado, para divisões categóricas. |

O scikit-learn não oferece ID3, C4.5 ou CHAID diretamente em `DecisionTreeClassifier`. O que usamos aqui é a implementação CART-like do scikit-learn.

## Hiperparâmetros Da Árvore No Scikit-Learn

| Hiperparâmetro | O que altera |
|---|---|
| `criterion` | Mede a qualidade da divisão. Pode ser `gini`, `entropy` ou `log_loss`. |
| `splitter` | Estratégia de escolha da divisão: `best` escolhe a melhor encontrada; `random` escolhe divisões aleatórias entre candidatas. |
| `max_depth` | Profundidade máxima da árvore. Limitar reduz overfitting. |
| `min_samples_split` | Número mínimo de amostras para dividir um nó. Valores maiores deixam a árvore mais conservadora. |
| `min_samples_leaf` | Número mínimo de amostras em uma folha. Aumentar suaviza a árvore. |
| `min_weight_fraction_leaf` | Fração mínima ponderada de amostras em uma folha. Útil com pesos. |
| `max_features` | Quantidade de atributos considerados em cada divisão. |
| `random_state` | Controla reprodutibilidade quando há aleatoriedade. |
| `max_leaf_nodes` | Limita o número de folhas. |
| `min_impurity_decrease` | Exige ganho mínimo de impureza para permitir uma divisão. |
| `class_weight` | Ajusta pesos das classes, útil em desbalanceamento. |
| `ccp_alpha` | Controla poda por custo-complexidade. Valores maiores podam mais. |
| `monotonic_cst` | Permite restrições monotônicas em atributos, quando aplicável. |

## Caminho B: Floresta Aleatória

Uma floresta aleatória combina várias árvores de decisão. Cada árvore é treinada com variações dos dados e dos atributos. Para classificar, as árvores votam e a floresta agrega os votos.

A diferença central em relação a uma árvore única é a redução de variância. Uma árvore sozinha pode mudar bastante se o treino mudar um pouco. A floresta reduz esse problema porque combina muitas árvores diferentes.

## Diferença Entre Floresta Aleatória E Árvore Única

| Aspecto | Árvore de decisão | Floresta aleatória |
|---|---|---|
| Quantidade de modelos | Uma árvore. | Muitas árvores. |
| Interpretabilidade | Alta. | Menor, porque há muitas árvores. |
| Variância | Maior. | Menor, por agregação. |
| Overfitting | Mais suscetível. | Geralmente mais robusta. |
| Custo computacional | Menor. | Maior. |
| Predição | Uma árvore decide. | Votação ou média das árvores. |

O `RandomForestClassifier` usa duas fontes principais de diversidade:

- bootstrap: cada árvore treina em uma amostra com reposição dos dados;
- subconjunto de atributos: em cada divisão, só parte das variáveis é considerada, exceto quando `max_features=None`.

## Algoritmo Usado Na Floresta

O código final usa os parâmetros salvos pelo Optuna:

```python
RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    n_estimators=216,
    min_samples_split=26,
    min_samples_leaf=11,
    max_features=None,
)
```

Isso significa:

- 216 árvores no conjunto;
- aleatoriedade reprodutível no modelo;
- paralelização usando todos os núcleos disponíveis;
- árvores mais suavizadas por `min_samples_split=26` e `min_samples_leaf=11`;
- `max_features=None`, ou seja, cada divisão pode considerar todos os atributos disponíveis.

## Algoritmos Relacionados A Florestas

Na biblioteca usada, o algoritmo diretamente usado é `RandomForestClassifier`. Existem modelos próximos, mas com diferenças:

| Modelo | Diferença |
|---|---|
| `RandomForestClassifier` | Treina árvores com bootstrap e seleção aleatória de atributos, conforme `max_features`. |
| `ExtraTreesClassifier` | Também usa várias árvores, mas escolhe divisões de forma mais aleatória. Pode reduzir variância ainda mais, mas aumenta viés. |
| Bagging com árvores | Agrega várias árvores por amostragem, mas sem necessariamente sortear atributos como a floresta aleatória padrão. |

No projeto, o Caminho B usa a floresta aleatória clássica via `RandomForestClassifier`.

## Hiperparâmetros Da Floresta No Scikit-Learn

| Hiperparâmetro | O que altera |
|---|---|
| `n_estimators` | Número de árvores. Mais árvores tendem a estabilizar o modelo, com maior custo. |
| `criterion` | Critério de divisão em cada árvore: `gini`, `entropy` ou `log_loss`. |
| `max_depth` | Profundidade máxima das árvores. |
| `min_samples_split` | Mínimo de amostras para dividir um nó. |
| `min_samples_leaf` | Mínimo de amostras em cada folha. |
| `min_weight_fraction_leaf` | Fração ponderada mínima em folha. |
| `max_features` | Número de atributos testados em cada divisão. |
| `max_leaf_nodes` | Limita folhas por árvore. |
| `min_impurity_decrease` | Ganho mínimo exigido para dividir. |
| `bootstrap` | Define se cada árvore usa amostra com reposição. |
| `oob_score` | Calcula desempenho out-of-bag quando `bootstrap=True`. |
| `n_jobs` | Número de processos paralelos. `-1` usa todos os núcleos. |
| `random_state` | Reprodutibilidade. |
| `verbose` | Nível de logs durante treino. |
| `warm_start` | Permite adicionar árvores a uma floresta já ajustada. |
| `class_weight` | Pesos por classe. |
| `ccp_alpha` | Poda por custo-complexidade em cada árvore. |
| `max_samples` | Quantidade ou fração de amostras usadas por árvore quando há bootstrap. |
| `monotonic_cst` | Restrições monotônicas, quando aplicáveis. |

## Caminho C: Naive Bayes Gaussiano

Naive Bayes é uma família de modelos probabilísticos baseada no Teorema de Bayes. A ideia é estimar a probabilidade de cada classe dado um conjunto de atributos.

O "naive" vem da suposição de independência condicional: dado o rótulo da classe, o modelo assume que os atributos são independentes entre si. Essa suposição raramente é perfeitamente verdadeira, mas o modelo pode funcionar bem como baseline.

## Variantes De Naive Bayes

O scikit-learn oferece várias versões:

| Variante | Quando faz sentido |
|---|---|
| `GaussianNB` | Atributos contínuos, assumindo distribuição normal por classe. |
| `MultinomialNB` | Contagens ou frequências não negativas, muito usado em texto. |
| `BernoulliNB` | Atributos binários, presença ou ausência de características. |
| `ComplementNB` | Variante do multinomial mais robusta para classes desbalanceadas em texto. |
| `CategoricalNB` | Atributos categóricos discretos codificados como categorias. |

O projeto usa `GaussianNB` porque, depois do pré-processamento, a matriz tem várias colunas numéricas. Porém, há uma ressalva: muitas colunas vêm de one-hot encoding, então nem tudo é realmente gaussiano. Por isso o Caminho C funciona mais como baseline comparativo do que como aposta principal.

## Como Funciona O `GaussianNB`

Para cada classe, o `GaussianNB` estima, em cada atributo:

- uma média;
- uma variância.

Ele assume que os valores daquele atributo seguem uma distribuição normal dentro de cada classe. Na predição, calcula a probabilidade dos atributos observados sob cada classe e escolhe a classe com maior probabilidade posterior.

O parâmetro otimizado pelo Optuna no Caminho C é:

```python
GaussianNB(var_smoothing=0.0009815405019274585)
```

`var_smoothing` adiciona uma pequena parcela à variância dos atributos. Isso evita instabilidade numérica quando alguma coluna tem variância muito pequena ou quase zero. No notebook, ele é buscado em escala logarítmica porque valores pequenos podem variar por ordens de grandeza.

## Hiperparâmetros De Naive Bayes No Scikit-Learn

Para o modelo usado, `GaussianNB`, os hiperparâmetros são:

| Hiperparâmetro | O que altera |
|---|---|
| `priors` | Probabilidades a priori das classes. Se `None`, são estimadas dos dados. |
| `var_smoothing` | Pequeno valor adicionado às variâncias para estabilidade numérica. |

Para as outras variantes, os hiperparâmetros comuns são:

| Variante | Hiperparâmetros principais |
|---|---|
| `MultinomialNB` | `alpha`, `force_alpha`, `fit_prior`, `class_prior`. |
| `BernoulliNB` | `alpha`, `force_alpha`, `binarize`, `fit_prior`, `class_prior`. |
| `ComplementNB` | `alpha`, `force_alpha`, `fit_prior`, `class_prior`, `norm`. |
| `CategoricalNB` | `alpha`, `force_alpha`, `fit_prior`, `class_prior`, `min_categories`. |

O significado geral:

- `alpha`: suavização para evitar probabilidades zero;
- `force_alpha`: controla se `alpha` é respeitado exatamente;
- `fit_prior`: define se as probabilidades das classes são aprendidas dos dados;
- `class_prior`: permite informar probabilidades de classe manualmente;
- `binarize`: limiar para converter atributos em binários no `BernoulliNB`;
- `norm`: normalização específica do `ComplementNB`;
- `min_categories`: número mínimo de categorias por atributo no `CategoricalNB`.

## Benchmarking

Depois de treinar e avaliar os três caminhos, o notebook cria uma tabela de ranking.

O ranking ordena por:

- F1;
- ROC AUC em caso de empate;
- acurácia em caso de novo empate.

O F1 é escolhido como critério principal porque equilibra precisão e recall. Isso é útil quando queremos evitar olhar apenas para acertos totais e perder a noção de como o modelo trata cada classe.

Na execução atual registrada nos artefatos de `20260605`, o ranking out-of-fold ficou:

| Posição | Caminho | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---:|---:|---:|---:|---:|
| 1 | Caminho B - Floresta Aleatória | 0.7988 | 0.8068 | 0.7896 | 0.7981 | 0.8866 |
| 2 | Caminho A - Árvore de Decisão | 0.7848 | 0.7681 | 0.8202 | 0.7933 | 0.8674 |
| 3 | Caminho C - Naive Bayes Gaussiano | 0.5809 | 0.5802 | 0.6071 | 0.5934 | 0.5876 |

O Caminho B fica em primeiro por F1 e também tem a maior ROC AUC. O Caminho A fica perto em F1, mas com ROC AUC menor. O Caminho C serve como baseline simples e mostra desempenho bem inferior neste pré-processamento.

## Mapa Entre EDA E Modelagem

| O que a EDA mostrou | Como o notebook usa |
|---|---|
| `PassengerId`, `Cabin` e `Name` têm informação composta. | Cria `PassengerGroup`, `CabinDeck`, `CabinSide` e `FamilyName`. |
| `HomePlanet` se relaciona com grupo e sobrenome. | Imputa `HomePlanet` por regras e depois pela moda global. |
| Gastos têm ausências e muitos zeros. | Imputa gastos individuais antes de criar `TotalSpend`. |
| `TotalSpend` resume consumo, mas gastos individuais têm nuances. | Usa gastos individuais e `TotalSpend`. |
| `CryoSleep` se relaciona com gastos e transporte. | Mantém `CryoSleep` como variável binária. |
| `CabinDeck` e `CabinSide` têm diferenças de taxa. | Mantém ambas no pré-processamento. |
| `Age` sugere corte não linear. | Usa modelos de árvore capazes de aprender limiares. |
| A escolha manual de hiperparâmetros pode ser limitada. | Usa Optuna para testar combinações e selecionar `best_params` por ROC AUC em validação cruzada. |

## Leitura Final

O `notebook.ipynb` é a etapa operacional do projeto. Ele pega as hipóteses da EDA, transforma essas hipóteses em uma pipeline reprodutível e compara três famílias de modelos.

O mais importante do notebook atual é que os modelos não são apenas instanciados com parâmetros fixos: cada caminho tem uma busca de hiperparâmetros com Optuna, os melhores parâmetros são persistidos em JSON e a avaliação final usa validação cruzada estratificada com predições out-of-fold. Isso deixa o fluxo mais sistemático e facilita reproduzir a comparação entre Árvore de Decisão, Floresta Aleatória e Naive Bayes Gaussiano.
