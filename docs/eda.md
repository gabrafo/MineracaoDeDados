# Explicação de Alto Nível do `eda.ipynb`

Este documento explica o que acontece em [`notebooks/eda.ipynb`](../notebooks/eda.ipynb). A EDA é a parte investigativa do projeto: ela entende os dados, procura padrões, identifica problemas de qualidade e levanta hipóteses para a modelagem.

O documento complementar é [`notebook.md`](./notebook.md), que explica como essas hipóteses viram uma pipeline de pré-processamento, validação cruzada e comparação de modelos.

## Ideia Central

O `eda.ipynb` responde perguntas como:

- quais colunas existem e o que elas significam;
- quais atributos têm valores ausentes;
- quais variáveis parecem se relacionar com `Transported`;
- quais colunas precisam ser quebradas em partes menores;
- quais decisões de pré-processamento parecem justificáveis.

Já o [`notebook.ipynb`](../notebooks/notebook.ipynb) pega essas respostas e transforma em código de modelagem. A relação entre os dois é proposital: a EDA dá o motivo, o notebook de modelagem implementa a decisão.

## Descrição Do Dataset

O notebook começa contextualizando a competição Spaceship Titanic. O objetivo é prever `Transported`, uma variável booleana que indica se o passageiro foi transportado para outra dimensão.

Os arquivos principais são:

- `train.csv`: dados de treino, com `Transported`.
- `test.csv`: dados de teste, sem `Transported`.

O dicionário de dados separa as variáveis em grupos:

- identificação e perfil: `PassengerId`, `Name`, `Age`;
- viagem: `HomePlanet`, `Destination`, `Cabin`, `CryoSleep`, `VIP`;
- gastos: `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`;
- alvo: `Transported`.

Essa explicação inicial é importante porque algumas colunas carregam informação composta. `PassengerId`, por exemplo, contém grupo de viagem e número do passageiro no grupo. `Cabin` contém convés, número e lado da cabine.

## Setup E Carregamento

A EDA importa bibliotecas de análise e visualização:

- `pandas` e `numpy` para trabalhar com tabelas e operações numéricas;
- `matplotlib` e `seaborn` para gráficos;
- `src.roots` para localizar os arquivos do projeto.

Depois, carrega:

- `train`, a partir de `data/train.csv`;
- `test`, a partir de `data/test.csv`.

Esse uso de `src.roots` é repetido no notebook de modelagem, mantendo os caminhos centralizados no projeto.

## Atributos Derivados Para Análise

A EDA cria uma função chamada `split_cabin_passenger`. Ela serve para transformar colunas compostas em atributos mais interpretáveis.

| Coluna original | Atributos derivados | Por que isso ajuda |
|---|---|---|
| `PassengerId` | `PassengerGroup`, `PassengerNum` | Permite analisar grupos de viagem. |
| `Cabin` | `CabinDeck`, `CabinNum`, `CabinSide` | Permite analisar convés e lado da cabine separadamente. |
| `Name` | `FamilyName` | Permite estudar relações familiares. |
| Gastos individuais | `TotalSpend` | Resume o gasto total conhecido do passageiro. |

Para a EDA, `TotalSpend` é calculado diretamente a partir dos gastos disponíveis, porque o objetivo ali é investigar padrões. No notebook de modelagem, a lógica foi refinada: se algum gasto individual estiver ausente, ele é imputado com KNN antes de calcular `TotalSpend`. Isso evita que `TotalSpend` vire uma soma parcial.

Também são criadas variáveis auxiliares para análise:

- `SpentSomething`: indica se o passageiro gastou algo em pelo menos um serviço.
- `HasMissingSpend`: indica se algum gasto está ausente.

## Entendimento Básico Dos Dados

O notebook verifica:

- tamanho do treino, do teste e da base enriquecida;
- tipos das colunas;
- estatísticas descritivas das variáveis numéricas;
- resumo das variáveis categóricas e booleanas;
- distribuição de `Transported`.

A leitura principal é que o treino tem 8.693 registros e o alvo está praticamente balanceado. Isso ajuda a justificar uma avaliação com várias métricas no notebook de modelagem: acurácia é informativa, mas F1, precisão, recall e ROC AUC dão uma visão mais completa.

## Valores Ausentes

A EDA calcula, para cada coluna:

- quantidade de valores preenchidos;
- quantidade de valores ausentes;
- percentual de ausência.

Também gera gráficos para visualizar a distribuição dos nulos.

Os percentuais de ausência são baixos, em torno de 2% por atributo, mas ainda precisam ser tratados. Essa observação leva diretamente à pipeline de modelagem, que usa imputadores para:

- `HomePlanet`, com regras por grupo e sobrenome e depois fallback pela moda global;
- gastos individuais, com KNN antes de criar `TotalSpend`;
- variáveis numéricas e binárias restantes;
- variáveis categóricas, com categoria explícita para ausentes.

## Relação Entre Atributos E `Transported`

Para variáveis categóricas de baixa cardinalidade, a EDA calcula a taxa de transporte por categoria:

- `HomePlanet`;
- `Destination`;
- `CryoSleep`;
- `VIP`;
- `CabinDeck`;
- `CabinSide`.

Para variáveis numéricas, calcula correlações com `Transported` e também correlações entre os próprios atributos.

O objetivo não é afirmar causalidade. A ideia é encontrar sinais úteis para modelagem. A partir disso, o notebook de modelagem mantém atributos como `CryoSleep`, `HomePlanet`, `Destination`, `CabinDeck` e `CabinSide`.

## Gastos E `TotalSpend`

A EDA analisa:

- `RoomService`;
- `FoodCourt`;
- `ShoppingMall`;
- `Spa`;
- `VRDeck`;
- `TotalSpend`.

Os gastos são esparsos, com muitos zeros. `TotalSpend` resume bem o comportamento geral de consumo, mas os gastos individuais não têm exatamente o mesmo sinal em relação ao alvo. `RoomService`, `Spa` e `VRDeck`, por exemplo, aparecem com relação mais negativa com `Transported` do que `FoodCourt` e `ShoppingMall`.

Por isso, a modelagem preserva as duas ideias:

- mantém os gastos individuais, porque eles carregam nuances diferentes;
- cria `TotalSpend`, porque o total agrega comportamento de consumo.

A melhoria aplicada no notebook de modelagem garante que `TotalSpend` seja calculado depois da imputação dos gastos individuais. Assim, quando há ausência em algum gasto, o total passa a representar a soma dos gastos conhecidos e imputados, não uma soma incompleta.

## Relação Entre `CryoSleep` E Gastos

A EDA verifica uma regra lógica do domínio: passageiros em `CryoSleep` deveriam gastar pouco ou nada, pois ficam confinados em suas cabines.

O notebook cruza `CryoSleep` com:

- taxa de transporte;
- mediana e média de `TotalSpend`;
- percentual de passageiros com algum gasto;
- percentual de passageiros com gasto ausente.

A conclusão é que `CryoSleep=True` se relaciona fortemente com gasto total baixo ou zero e com taxa maior de transporte. Por isso, `CryoSleep` é mantido como variável importante na modelagem.

## Grupo, Família E Planeta De Origem

A EDA explora a relação entre:

- `PassengerGroup`, extraído de `PassengerId`;
- `FamilyName`, extraído de `Name`;
- `HomePlanet`.

A hipótese é que pessoas viajando juntas ou com o mesmo sobrenome tendem a compartilhar origem. O notebook mede cardinalidade, tamanhos dos grupos, taxas de transporte e associação com `HomePlanet`.

Essa análise vira uma decisão concreta no notebook de modelagem: `HomePlanet` é imputado primeiro pela moda do `PassengerGroup`; se ainda faltar valor, pela moda do `FamilyName`; se ainda assim faltar, entra como fallback o valor mais frequente de `HomePlanet` no dataset de treino.

## Idade E Não Linearidade

A EDA analisa `Age` de duas formas:

- por faixas interpretáveis de idade;
- por limiares binários testados com impureza de Gini.

O melhor corte encontrado é `Age <= 4`, com taxa de transporte bem maior para crianças pequenas. Isso sugere uma relação não linear: a idade não parece atuar apenas como um número contínuo simples.

O notebook de modelagem não cria manualmente uma coluna `Age <= 4`, mas usa modelos baseados em árvores, que conseguem aprender cortes desse tipo automaticamente.

## Cabine: Convés E Lado

A EDA cruza:

- `CabinDeck`;
- `CabinSide`;
- `Transported`.

Ela calcula taxas de transporte e contagens por combinação, além de gerar um mapa de calor.

A conclusão é que faz sentido preservar `CabinDeck` e `CabinSide` no pré-processamento. No notebook de modelagem, `CabinDeck` entra como categórica multi-classe e `CabinSide` entra como variável binária codificada.

## Ponte Com A Modelagem

| Descoberta no `eda.ipynb` | Decisão no `notebook.ipynb` |
|---|---|
| `PassengerId`, `Cabin` e `Name` carregam informação composta. | `SpaceshipFeatureBuilder` cria grupo, cabine decomposta e sobrenome. |
| Há nulos em baixa proporção. | A pipeline trata ausências com regras, KNN e imputação categórica. |
| `HomePlanet` se relaciona com grupo e sobrenome. | `HomePlanetRuleImputer` usa `PassengerGroup` e `FamilyName`; nulos restantes usam a moda global. |
| Gastos individuais têm muitos zeros e ausências. | `SpendKNNImputerAndTotalSpend` imputa gastos antes de criar `TotalSpend`. |
| `TotalSpend` resume consumo, mas gastos individuais têm nuances. | O modelo usa gastos individuais e `TotalSpend`. |
| `CryoSleep` tem relação forte com gastos e alvo. | `CryoSleep` é mantido e codificado na pipeline. |
| Idade mostra sinal não linear. | Árvores e florestas podem aprender cortes de idade automaticamente. |
| `CabinDeck` e `CabinSide` têm diferenças de taxa. | As duas variáveis são preservadas no pré-processamento. |

Depois dessas decisões de pré-processamento, o notebook de modelagem atual usa Optuna para buscar hiperparâmetros dos três caminhos e comparar os modelos com validação cruzada estratificada. Assim, a EDA orienta quais sinais entram na pipeline, enquanto o Optuna ajuda a escolher configurações melhores para cada algoritmo.

## Leitura Final

O `eda.ipynb` é o notebook que dá sentido às escolhas do projeto. Ele não é apenas uma coleção de gráficos: ele cria a base lógica para o pré-processamento, para a escolha dos atributos e para a comparação dos modelos descritos em [`notebook.md`](./notebook.md).
